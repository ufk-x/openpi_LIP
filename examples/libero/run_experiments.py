#!/usr/bin/env python3
"""
LIBERO 批量实验编排器
====================
只加载模型一次，循环多组 RTC 参数进行评估。

用法：
    cd /home/ps/Projects/openpi
    uv run examples/libero/run_experiments.py

流程（每组参数）：
    1. 修改模型实例属性（零开销）
    2. 在后台线程启动 WebSocket 服务
    3. 用子进程运行客户端评估 (examples/libero/main.py)
    4. 客户端结束后优雅关闭服务
    5. 进入下一组
"""

import asyncio
import dataclasses
import logging
import os
import pathlib
import subprocess
import sys
import threading
import time

# ============================================================
# ★ 实验参数配置区
# ============================================================

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # openpi/
# 客户端使用独立的 Python 3.8 虚拟环境（参见 README "不使用 Docker" 部分）
CLIENT_PYTHON = str(PROJECT_ROOT / "examples" / "libero" / ".venv" / "bin" / "python")

@dataclasses.dataclass
class ExperimentConfig:
    """单组实验参数。"""
    rtc_flag: bool
    replan_steps: int
    delay_steps: int
    schedule: str = "exp"
    max_weight: float = 5.0

# ---- 在这里添加/删除要测试的参数组合 ----
EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(rtc_flag=True,  replan_steps=5, delay_steps=1),
    ExperimentConfig(rtc_flag=False,  replan_steps=5, delay_steps=1),
    ExperimentConfig(rtc_flag=True,  replan_steps=5, delay_steps=2),
    ExperimentConfig(rtc_flag=False,  replan_steps=5, delay_steps=2),
    ExperimentConfig(rtc_flag=True,  replan_steps=5, delay_steps=3),
    ExperimentConfig(rtc_flag=False,  replan_steps=5, delay_steps=3),
    ExperimentConfig(rtc_flag=True,  replan_steps=5, delay_steps=4),
    ExperimentConfig(rtc_flag=False,  replan_steps=5, delay_steps=4),
    ExperimentConfig(rtc_flag=True,  replan_steps=5, delay_steps=5),
    ExperimentConfig(rtc_flag=False,  replan_steps=5, delay_steps=5),
]

# ---- 固定参数 ----
PORT = 8000
TASK_SUITE = "libero_10" # 评测哪个任务集（ libero_spatial, libero_object, libero_goal, libero_10, libero_90 ）
NUM_TRIALS = 10

# ============================================================
# 以下无需修改
# ============================================================

logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("run_experiments")


def load_policy_once():
    """加载模型（只调用一次，耗时约 1-2 分钟）。"""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    logger.info("========== 加载模型（仅此一次）==========")
    config = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(
        config, "gs://openpi-assets/checkpoints/pi05_libero"
    )
    logger.info("========== 模型加载完成 ==========")
    return policy


def apply_rtc_config(policy, exp: ExperimentConfig):
    """将实验参数写入模型实例属性（零开销）。"""
    model = policy._model
    model.rtc_guidance = exp.rtc_flag
    model.replan_steps = exp.replan_steps
    model.guidance_inference_delay = exp.delay_steps
    model.guidance_prefix_attention_horizon = model.action_horizon - exp.replan_steps
    model.guidance_prefix_attention_schedule = exp.schedule
    model.guidance_max_weight = exp.max_weight
    # 每组实验开始时重置 prev_action_chunk
    model.prev_action_chunk = None
    logger.info(
        "RTC 参数已设置: rtc=%s, replan=%d, delay=%d, horizon=%d, schedule=%s, max_weight=%.1f",
        exp.rtc_flag, exp.replan_steps, exp.delay_steps,
        model.action_horizon - exp.replan_steps, exp.schedule, exp.max_weight,
    )


def run_server_in_thread(policy, port: int):
    """在后台线程中启动 WebSocket 服务，返回 (thread, stop_event, server_ready_event)。

    直接复用 WebsocketPolicyServer.run()，保证与 serve_policy.py 完全一致
    （包括 _health_check、compression 设置等）。
    """
    from openpi.serving import websocket_policy_server

    stop_event = threading.Event()
    ready_event = threading.Event()

    ws_server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata=policy.metadata,
    )

    async def _serve():
        import websockets.asyncio.server as _ws_server
        from openpi.serving.websocket_policy_server import _health_check

        async with _ws_server.serve(
            ws_server._handler,
            ws_server._host,
            ws_server._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            ready_event.set()
            logger.info("WebSocket 服务已启动 (端口 %d)", port)

            # 等待停止信号
            while not stop_event.is_set():
                await asyncio.sleep(0.5)

            server.close()
            await server.wait_closed()
            logger.info("WebSocket 服务已关闭")

    def _thread_target():
        asyncio.run(_serve())

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    return t, stop_event, ready_event


def run_client(exp: ExperimentConfig, port: int):
    """以子进程方式运行评估客户端。

    使用 examples/libero/.venv 的 Python 3.8 环境（与 README "不使用 Docker" 一致），
    并设置 PYTHONPATH 包含 third_party/libero。
    """
    # tyro.cli(eval_libero) 的签名是 eval_libero(args: Args)，
    # 因此所有 CLI 参数需要加 --args. 前缀。
    cmd = [
        CLIENT_PYTHON, str(PROJECT_ROOT / "examples" / "libero" / "main.py"),
        "--args.host", "127.0.0.1",
        "--args.port", str(port),
        "--args.task-suite-name", TASK_SUITE,
        "--args.num-trials-per-task", str(NUM_TRIALS),
        "--args.replan-steps", str(exp.replan_steps),
        "--args.delay-steps", str(exp.delay_steps),
        "--args.guidance-prefix-attention-schedule", exp.schedule,
        "--args.guidance-max-weight", str(exp.max_weight),
    ]
    if exp.rtc_flag:
        cmd.append("--args.rtc-flag")
    else:
        cmd.append("--args.no-rtc-flag")

    # 设置环境变量：PYTHONPATH 需要包含 third_party/libero（README 要求）
    env = os.environ.copy()
    libero_path = str(PROJECT_ROOT / "third_party" / "libero")
    env["PYTHONPATH"] = libero_path + os.pathsep + env.get("PYTHONPATH", "")

    logger.info("启动客户端: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        logger.warning("客户端退出码: %d", result.returncode)
    return result.returncode


def main():
    total = len(EXPERIMENTS)
    logger.info("共 %d 组实验", total)
    for i, exp in enumerate(EXPERIMENTS):
        logger.info("  [%d/%d] rtc=%s  replan=%d  delay=%d", i + 1, total, exp.rtc_flag, exp.replan_steps, exp.delay_steps)

    # ---- 只加载一次模型 ----
    policy = load_policy_once()

    for i, exp in enumerate(EXPERIMENTS):
        logger.info("")
        logger.info("=" * 60)
        logger.info("  实验 %d/%d: rtc=%s  replan=%d  delay=%d", i + 1, total, exp.rtc_flag, exp.replan_steps, exp.delay_steps)
        logger.info("=" * 60)

        # 1. 设置本轮参数
        apply_rtc_config(policy, exp)

        # 2. 启动服务
        server_thread, stop_event, ready_event = run_server_in_thread(policy, PORT)
        if not ready_event.wait(timeout=30):
            logger.error("服务端启动超时，跳过本轮")
            stop_event.set()
            server_thread.join(timeout=10)
            continue

        # 给服务一点缓冲时间
        time.sleep(1)

        # 3. 运行客户端
        try:
            run_client(exp, PORT)
        except Exception as e:
            logger.error("客户端异常: %s", e)

        # 4. 停止服务
        logger.info("关闭服务端...")
        stop_event.set()
        server_thread.join(timeout=15)
        time.sleep(1)  # 等端口释放

        logger.info("实验 %d/%d 完成 ✓", i + 1, total)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  全部 %d 组实验完成！", total)
    logger.info("  结果: data/libero/eval_results_delay*_replan*_rtc*.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
