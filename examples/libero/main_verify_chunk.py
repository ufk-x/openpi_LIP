import collections
import dataclasses
import logging
import math
import pathlib

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from typing import List, Optional

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
CHUNK_SIZE = 10  # 用于 realtime guidance 的 action chunk 大小

RTC_FLAG: bool = False  # 是否启用 realtime guidance
DELAY_STEPS: int = 3  # 推理延迟步数
REPLANE_STEPS: int = 5  # 每次 replanning 的执行步数（execute_horizon）
GUIDANCE_PREFIX_ATTENTION_SCHEDULE = "exp"  # prefix attention 调度方式
GUIDANCE_MAX_WEIGHT: float = 5.0  # RTC guidance 权重的默认值

@dataclasses.dataclass
class Args:
    # 模型服务
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    # LIBERO 任务
    task_suite_name: str = "libero_10"
    task_id: int = 0
    episode_idx: int = 0
    num_steps_wait: int = 10
    max_infer_n: int = 10  # 冻结前最多进行的 infer 次数（不含冻结后的额外 infer）

    # 可视化
    action_dim_to_plot: int = 0
    num_new_chunks: int = 5
    max_chunk_points: int = -1  # <=0 表示不截断
    plot_out_path: str = ""

    # rtc 参数（会传递到模型实例属性）
    rtc: bool = RTC_FLAG  # 是否启用 realtime guidance
    replan_steps: int = REPLANE_STEPS  # 每次 replanning 的执行步数（execute_horizon）
    delay_steps: int = DELAY_STEPS  # 推理延迟步数
    guidance_prefix_attention_schedule: str = GUIDANCE_PREFIX_ATTENTION_SCHEDULE  # prefix attention 调度方式
    guidance_max_weight: float = GUIDANCE_MAX_WEIGHT  # RTC guidance 权重的最大值

    # 复现
    seed: int = 42


def _prepare_obs(obs, resize_size: int):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def _build_policy_input(
    img,
    wrist_img,
    obs,
    task_description: str,
    rtc_guidance_chunk: Optional[np.ndarray] = None,
    rtc_config: Optional[dict] = None,
) -> dict:
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }
    if rtc_guidance_chunk is not None:
        element["rtc_guidance_chunk"] = np.asarray(rtc_guidance_chunk)
    if rtc_config is not None:
        element.update(rtc_config)
    return element


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _infer_chunk(
    client,
    obs,
    task_description: str,
    resize_size: int,
    rtc_guidance_chunk: Optional[np.ndarray] = None,
    rtc_config: Optional[dict] = None,
) -> np.ndarray:
    img, wrist_img = _prepare_obs(obs, resize_size)
    element = _build_policy_input(
        img,
        wrist_img,
        obs,
        task_description,
        rtc_guidance_chunk=rtc_guidance_chunk,
        rtc_config=rtc_config,
    )
    chunk = np.asarray(client.infer(element)["actions"])
    if chunk.ndim != 2:
        raise ValueError(f"模型返回的 actions 维度异常: shape={chunk.shape}，期望 [horizon, action_dim]")
    return chunk


def _maybe_truncate(chunk: np.ndarray, max_chunk_points: int) -> np.ndarray:
    if max_chunk_points is None or max_chunk_points <= 0:
        return chunk
    return chunk[:max_chunk_points]


def verify_chunk(args: Args) -> None:
    np.random.seed(args.seed)

    if args.max_infer_n <= 0:
        raise ValueError(f"max_infer_n 必须 > 0，当前值: {args.max_infer_n}")
    if args.num_new_chunks <= 0:
        raise ValueError(f"num_new_chunks 必须 > 0，当前值: {args.num_new_chunks}")
    if args.delay_steps < 0:
        raise ValueError(f"delay_steps 必须 >= 0，当前值: {args.delay_steps}")
    if args.replan_steps <= 0:
        raise ValueError(f"replan_steps 必须 > 0，当前值: {args.replan_steps}")
    if CHUNK_SIZE - args.replan_steps < args.delay_steps:
        raise ValueError(
            f"prefix_size[CHUNK_SIZE - replan_steps ({CHUNK_SIZE - args.replan_steps})] "
            f"必须 >= delay_steps ({args.delay_steps})"
        )
    use_delay = args.delay_steps > 0

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    task_suite = benchmark_dict[args.task_suite_name]()
    if args.task_id < 0 or args.task_id >= task_suite.n_tasks:
        raise ValueError(f"task_id 越界: {args.task_id}，有效范围 [0, {task_suite.n_tasks - 1}]")

    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    if args.episode_idx < 0 or args.episode_idx >= len(initial_states):
        raise ValueError(f"episode_idx 越界: {args.episode_idx}，有效范围 [0, {len(initial_states) - 1}]")

    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    if not args.plot_out_path:
        args.plot_out_path = (
            f"data/libero/chunk_verify_{args.task_suite_name}_task{args.task_id}_"
            f"ep{args.episode_idx}_infer{args.max_infer_n}_dim{args.action_dim_to_plot}_rtc{args.rtc}_replan{args.replan_steps}_delay{args.delay_steps}_schedule{args.guidance_prefix_attention_schedule}.png"
        )
    pathlib.Path(args.plot_out_path).parent.mkdir(parents=True, exist_ok=True)

    pre_chunk: Optional[np.ndarray] = None
    done = False
    env_timestamp = 0
    rtc_infer_config = {
        "rtc": args.rtc,
        "replan_steps": args.replan_steps,
        "delay_steps": args.delay_steps,
        "guidance_prefix_attention_schedule": args.guidance_prefix_attention_schedule,
        "guidance_max_weight": args.guidance_max_weight,
    }

    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_idx])

        action_plan = collections.deque()
        prev_chunk: Optional[np.ndarray] = None
        prev_chunk_offset: int = 0
        infer_count = 0

        while not done:
            if env_timestamp < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                env_timestamp += 1
                continue

            if not action_plan:
                if infer_count >= args.max_infer_n-1:  # 先进行 max_infer_n-1 次 infer
                    break
                rtc_infer_config["rtc"] = False # 冻结前的infer都不启用RTC
                new_chunk = _infer_chunk(
                    client,
                    obs,
                    task_description,
                    args.resize_size,
                    rtc_guidance_chunk=prev_chunk if (rtc_infer_config["rtc"] and prev_chunk is not None) else None,
                    rtc_config=rtc_infer_config,
                ) # rtc_guidance_chunk的传递由rtc_infer_config["rtc"]控制
                infer_count += 1
                if len(new_chunk) < args.replan_steps:
                    raise ValueError(
                        f"策略仅预测了 {len(new_chunk)} 步，至少需要 {args.replan_steps} 步。"
                    )

                if use_delay and prev_chunk is not None:
                    delay_actions = []
                    for d in range(args.delay_steps):
                        idx = prev_chunk_offset + d
                        if idx < len(prev_chunk):
                            delay_actions.append(prev_chunk[idx])
                        else:
                            delay_actions.append(prev_chunk[-1])
                    new_start = args.delay_steps
                    new_end = args.replan_steps
                    combined = delay_actions + list(new_chunk[new_start:new_end])
                    action_plan.extend(combined)
                else:
                    action_plan.extend(new_chunk[: args.replan_steps])

                prev_chunk = new_chunk
                prev_chunk_offset = args.replan_steps

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist() if isinstance(action, np.ndarray) else action)
            env_timestamp += 1

        # 正常完成 max_infer_n-1 次 infer 后，进行最后一次 infer。
        pre_chunk_timestamp = env_timestamp # pre_chunk 的时间戳为本次 infer 时的环境时间戳
        rtc_infer_config["rtc"] = False # 冻结前的infer都不启用RTC
        pre_chunk = _infer_chunk(
            client,
            obs,
            task_description,
            args.resize_size,
            rtc_guidance_chunk=prev_chunk if (rtc_infer_config["rtc"] and prev_chunk is not None) else None,
            rtc_config=rtc_infer_config,
        ) # rtc_guidance_chunk的传递由rtc_infer_config["rtc"]控制

        if len(pre_chunk) < args.replan_steps:
            raise ValueError(
                f"最后一次 infer 仅预测了 {len(pre_chunk)} 步，至少需要 {args.replan_steps} 步用于冻结前执行。"
            )

        # 最后一次 infer 后，环境继续执行 replan_steps 步，再冻结环境状态。
        for step_idx in range(args.replan_steps):
            if done:
                break
            action = pre_chunk[step_idx]
            obs, _, done, _ = env.step(action.tolist() if isinstance(action, np.ndarray) else action)
            env_timestamp += 1

        # 到这里冻结环境：不再调用 env.step，后续只做并行时间 infer。
        freeze_timestamp = env_timestamp # new chunks的时间戳从 freeze_timestamp开始

        if args.action_dim_to_plot < 0 or args.action_dim_to_plot >= pre_chunk.shape[1]:
            raise ValueError(
                f"action_dim_to_plot 越界: {args.action_dim_to_plot}，模型动作维度为 {pre_chunk.shape[1]}"
            )

        # pre_chunk_plot = _maybe_truncate(pre_chunk, args.max_chunk_points)
        x_pre = np.arange(pre_chunk_timestamp, pre_chunk_timestamp + len(pre_chunk)) # pre_chunk 的 x 轴坐标从 pre_chunk_timestamp 开始，长度为 pre_chunk_plot 的长度

        new_chunks_no_rtc = []
        rtc_infer_config["rtc"] = False
        for _ in range(args.num_new_chunks):
            new_chunk = _infer_chunk(
                client,
                obs,
                task_description,
                args.resize_size,
                rtc_guidance_chunk=None,
                rtc_config=rtc_infer_config,
            )
            if new_chunk.shape[1] != pre_chunk.shape[1]:
                raise ValueError(
                    f"new_chunk(no_rtc) 动作维度 ({new_chunk.shape[1]}) 与 pre_chunk ({pre_chunk.shape[1]}) 不一致"
                )
            new_chunks_no_rtc.append(new_chunk)

        new_chunks_rtc = []
        rtc_infer_config["rtc"] = True
        for _ in range(args.num_new_chunks):
            new_chunk = _infer_chunk(
                client,
                obs,
                task_description,
                args.resize_size,
                rtc_guidance_chunk=pre_chunk,
                rtc_config=rtc_infer_config,
            )
            if new_chunk.shape[1] != pre_chunk.shape[1]:
                raise ValueError(
                    f"new_chunk(rtc) 动作维度 ({new_chunk.shape[1]}) 与 pre_chunk ({pre_chunk.shape[1]}) 不一致"
                )
            new_chunks_rtc.append(new_chunk)

        delay_line_x = freeze_timestamp + args.delay_steps - 1

        def _mean_l2_distance_in_delay_window(chunks: List[np.ndarray]) -> float:
            if args.delay_steps <= 0:
                return float("nan")

            pre_start = freeze_timestamp - pre_chunk_timestamp
            pre_end = delay_line_x - pre_chunk_timestamp
            if pre_end < pre_start:
                return float("nan")

            dists = []
            for chunk in chunks:
                new_start = 0
                new_end = args.delay_steps - 1

                valid_start = max(pre_start, 0)
                valid_end = min(pre_end, len(pre_chunk) - 1)
                if valid_end < valid_start:
                    continue

                span = valid_end - valid_start + 1
                if new_start + span - 1 > new_end or new_start + span > len(chunk):
                    span = min(new_end - new_start + 1, len(chunk), span)
                if span <= 0:
                    continue

                pre_seg = pre_chunk[valid_start : valid_start + span, args.action_dim_to_plot]
                new_seg = chunk[new_start : new_start + span, args.action_dim_to_plot]
                diff = new_seg - pre_seg
                dists.append(float(np.sqrt(np.sum(np.square(diff)))))

            if not dists:
                return float("nan")
            return float(np.mean(dists))

        mean_dist_no_rtc = _mean_l2_distance_in_delay_window(new_chunks_no_rtc)
        mean_dist_rtc = _mean_l2_distance_in_delay_window(new_chunks_rtc)
        improvement = (mean_dist_no_rtc - mean_dist_rtc) / mean_dist_no_rtc * 100 if not math.isnan(mean_dist_no_rtc) and not math.isnan(mean_dist_rtc) else float("nan")


        plt.figure(figsize=(10, 5))
        plt.plot(
            x_pre,
            pre_chunk[:, args.action_dim_to_plot],
            label="pre_chunk(no rtc)",
            linewidth=2.5,
            color="black",
        )

        for idx, chunk in enumerate(new_chunks_no_rtc):
            x_new = np.arange(freeze_timestamp, freeze_timestamp + len(chunk)) # new_chunk 的 x 轴坐标从 freeze_timestamp 开始，长度为 chunk 的长度
            plt.plot(
                x_new,
                chunk[:, args.action_dim_to_plot],
                label="new_chunk_no_rtc" if idx == 0 else None,
                linewidth=1.6,
                alpha=0.5,
                color="red",
            )

        for idx, chunk in enumerate(new_chunks_rtc):
            x_new = np.arange(freeze_timestamp, freeze_timestamp + len(chunk)) # new_chunk 的 x 轴坐标从 freeze_timestamp 开始，长度为 chunk 的长度
            plt.plot(
                x_new,
                chunk[:, args.action_dim_to_plot],
                label="new_chunk_rtc" if idx == 0 else None,
                linewidth=1.6,
                alpha=0.9,
                color="green",
            )

        # 描绘delay的竖直线
        if use_delay:
            plt.axvline(x=delay_line_x, color="red", linestyle="--", label="delay boundary")

        if use_delay:
            dist_text = (
                f"mean L2(pre,new) [{freeze_timestamp},{delay_line_x}]\n"
                f"no_rtc: {mean_dist_no_rtc:.6f}\n"
                f"rtc: {mean_dist_rtc:.6f}"
                f"\nimprovement: {improvement:.2f}%"
            )
        else:
            dist_text = "mean L2(pre,new) [freeze, delay_line]: N/A (delay_steps<=0)"
        plt.gca().text(
            0.02,
            0.02,
            dist_text,
            transform=plt.gca().transAxes,
            verticalalignment="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        )

        plt.xlabel("env timestamp")
        plt.ylabel(f"action[{args.action_dim_to_plot}]")
        plt.title(
            f"rtc_on+off|delay_steps={args.delay_steps}|replan_steps={args.replan_steps}|suite={args.task_suite_name}|task={args.task_id}|"
            f"ep={args.episode_idx}|freeze_t={freeze_timestamp}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_out_path, dpi=180)
        plt.close()

        logging.info(f"Task description: {task_description}")
        logging.info(f"Environment frozen at timestamp: {freeze_timestamp}")
        logging.info(f"pre_chunk shape: {pre_chunk.shape}")
        logging.info(f"num_new_chunks_no_rtc: {len(new_chunks_no_rtc)}")
        logging.info(f"num_new_chunks_rtc: {len(new_chunks_rtc)}")
        logging.info(f"mean_l2_distance_no_rtc_in_delay_window: {mean_dist_no_rtc}")
        logging.info(f"mean_l2_distance_rtc_in_delay_window: {mean_dist_rtc}")
        logging.info(f"improvement: {improvement:.2f}%")
        logging.info(f"Plot saved to: {args.plot_out_path}")

    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(verify_chunk)
