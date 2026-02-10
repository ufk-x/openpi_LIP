import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
import pandas as pd
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # 渲染训练数据使用的分辨率

CHUNK_SIZE = 10  # 用于 realtime guidance 的 action chunk 大小
REPLAN_STEPS = 2  # 每次 replanning 时的重叠步数，论文中的execute_horizon
DELAY_STEPS = 1 # 严格要求： DELAY_STEPS <= CHUNK_SIZE - REPLAN_STEPS （prefix_attention_horizon）
RTC_GUIDANCE = True  # 是否启用 realtime guidance

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # 模型服务器参数
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = REPLAN_STEPS # action chunk只有10步，replan_steps需要小于等于这个值

    #################################################################################################################
    # LIBERO 环境参数
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # 任务套件。可选: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # 等待物体在仿真中稳定的步数
    num_trials_per_task: int = 10  # 每个任务的评估回合数

    #################################################################################################################
    # 推理延迟参数（参考 eval_flow.py）
    #################################################################################################################
    inference_delay: int = DELAY_STEPS  # 模拟推理延迟的环境步数（0 = 无延迟）
    # 要求：replan_steps + inference_delay <= action chunk(10)

    #################################################################################################################
    # 工具参数
    #################################################################################################################
    video_out_path: str = "data/libero/videos"+str(task_suite_name)  # 视频保存路径
    delay_video_out_path: str = "data/libero/delay_videos"+str(task_suite_name)  # 带模拟推理延迟的视频保存路径
    csv_out_path: str = f"data/libero/eval_results_delay{inference_delay}_replan{replan_steps}_rtc{RTC_GUIDANCE}.csv"  # 评估结果 CSV 保存路径

    seed: int = 7  # 随机种子（用于复现）


def _get_max_steps(task_suite_name: str) -> int:
    """返回指定任务套件的最大步数。"""
    max_steps_map = {
        "libero_spatial": 220,   # 最长训练演示有 193 步
        "libero_object": 280,   # 最长训练演示有 254 步
        "libero_goal": 300,     # 最长训练演示有 270 步
        "libero_10": 520,       # 最长训练演示有 505 步
        "libero_90": 400,       # 最长训练演示有 373 步
    }
    if task_suite_name not in max_steps_map:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return max_steps_map[task_suite_name]


def _prepare_obs(obs, resize_size: int):
    """预处理观测数据：旋转 180 度并填充缩放。"""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    return img, wrist_img


def _build_policy_input(img, wrist_img, obs, task_description: str) -> dict:
    """构建用于策略推理的观测字典。"""
    return {
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


def eval_libero(args: Args) -> None:
    """评估 LIBERO 任务，支持可选的模拟推理延迟。

    当 inference_delay > 0 时，评估会模拟推理缓慢的效果：
    - 每个重规划周期共执行 `replan_steps` 个动作（作为 execute_horizon）。
    - 前 `inference_delay` 个动作来自上一轮 action chunk（旧计划），
      模拟新的 action chunk 尚未计算完成。
    - 剩余 `replan_steps - inference_delay` 个动作来自新的 action chunk
      （从索引 `inference_delay` 开始），与 eval_flow.py 的模式一致。
    - 延迟评估的视频保存到 `delay_video_out_path`。

    指标（每个任务的平均回合长度、累计奖励、成功率）输出到 `csv_out_path` 的 CSV 文件。
    """
    # 设置随机种子
    np.random.seed(args.seed)

    # 验证延迟参数
    assert args.inference_delay >= 0, f"inference_delay 必须 >= 0，当前值: {args.inference_delay}"
    # assert args.replan_steps > args.inference_delay, (
    #     f"replan_steps ({args.replan_steps}) 必须 > inference_delay ({args.inference_delay})"
    # )
    assert CHUNK_SIZE - args.replan_steps >= args.inference_delay, (
        f"prefix_size[CHUNK_SIZE - replan_steps ({CHUNK_SIZE - args.replan_steps})] 必须 >= inference_delay ({args.inference_delay})"
    )
    use_delay = args.inference_delay > 0
    video_dir = args.delay_video_out_path if use_delay else args.video_out_path

    # 初始化 LIBERO 任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = _get_max_steps(args.task_suite_name)
    logging.info(f"Task suite: {args.task_suite_name} | inference_delay: {args.inference_delay}")

    pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.csv_out_path).parent.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # ---- 指标收集 ----
    csv_rows = []  # 存储每个任务的聚合指标
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        task_episode_lengths = []
        task_cumulative_rewards = []

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # 重置环境
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            cumulative_reward = 0.0

            # ---- 动作块管理 ----
            # `action_plan`       : 当前重规划周期中待执行的动作队列
            # `prev_chunk`        : 上一轮完整的 action chunk（用于延迟时提供「旧」动作）
            # `prev_chunk_offset` : prev_chunk 中下一个延迟动作的指针位置
            action_plan: collections.deque = collections.deque()
            prev_chunk: np.ndarray | None = None
            prev_chunk_offset: int = 0

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # 等待阶段：让物体稳定
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # 预处理观测
                    img, wrist_img = _prepare_obs(obs, args.resize_size)
                    replay_images.append(img)

                    # ---- 当动作队列耗尽时重新规划 ----
                    if not action_plan:
                        element = _build_policy_input(img, wrist_img, obs, task_description)

                        # 查询模型获取新的动作块
                        new_chunk = np.array(client.infer(element)["actions"])
                        assert len(new_chunk) >= args.replan_steps, (
                            f"策略仅预测了 {len(new_chunk)} 步，至少需要 {args.replan_steps} 步。"
                        )

                        if use_delay and prev_chunk is not None:
                            # 模拟推理延迟模式（eval_flow.py 风格）：
                            #   前 `inference_delay` 个动作 -> 来自上一轮 chunk（「旧」动作）
                            #   后 `replan_steps - inference_delay` 个动作 -> 来自新 chunk
                            delay_actions = []
                            for d in range(args.inference_delay):
                                idx = prev_chunk_offset + d
                                if idx < len(prev_chunk):
                                    delay_actions.append(prev_chunk[idx])
                                else:
                                    # 上一轮 chunk 已耗尽，重复最后一个可用动作
                                    delay_actions.append(prev_chunk[-1])
                            new_start = args.inference_delay
                            new_end = args.replan_steps
                            combined = delay_actions + list(new_chunk[new_start:new_end])
                            action_plan.extend(combined)
                        else:
                            # 无延迟 或 第一个 chunk（没有上一轮可用）
                            action_plan.extend(new_chunk[: args.replan_steps])

                        # 保存当前 chunk 作为下一轮重规划的旧 chunk
                        prev_chunk = new_chunk
                        prev_chunk_offset = args.replan_steps  # 旧 chunk 在 execute_horizon 之后的续接位置

                    # 执行一个动作
                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(
                        action.tolist() if isinstance(action, np.ndarray) else action
                    )
                    cumulative_reward += reward

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            episode_length = t - args.num_steps_wait  # 有效步数（排除等待阶段）
            task_episode_lengths.append(episode_length)
            task_cumulative_rewards.append(cumulative_reward)
            task_episodes += 1
            total_episodes += 1

            # 保存回放视频
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            delay_tag = f"_delay{args.inference_delay}" if use_delay else ""
            replan_tag = f"_replan{args.replan_steps}"
            imageio.mimwrite(
                pathlib.Path(video_dir) / f"rollout_rtc{RTC_GUIDANCE}_{task_segment}{delay_tag}{replan_tag}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # ---- 单任务指标 ----
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        avg_episode_length = float(np.mean(task_episode_lengths)) if task_episode_lengths else 0.0
        avg_cumulative_reward = float(np.mean(task_cumulative_rewards)) if task_cumulative_rewards else 0.0

        csv_rows.append({
            "task_suite": args.task_suite_name,
            "task_id": task_id,
            "task_description": task_description,
            "inference_delay": args.inference_delay,
            "replan_steps": args.replan_steps,
            "num_episodes": task_episodes,
            "successes": task_successes,
            "success_rate": task_success_rate,
            "avg_episode_length": avg_episode_length,
            "avg_cumulative_reward": avg_cumulative_reward,
        })

        logging.info(f"Task '{task_description}' => success_rate={task_success_rate:.3f}, "
                     f"avg_len={avg_episode_length:.1f}, avg_reward={avg_cumulative_reward:.3f}")

        # 显式关闭环境，避免退出时 EGL 清理报错
        env.close()

    # ---- 写入 CSV ----
    df = pd.DataFrame(csv_rows)
    df.to_csv(args.csv_out_path, index=False)
    logging.info(f"Results saved to {args.csv_out_path}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """初始化并返回 LIBERO 环境及任务描述。"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # 重要：即使使用固定初始状态，种子也会影响物体位置
    return env, task_description


def _quat2axisangle(quat):
    """
    从 robosuite 复制: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # 裁剪四元数
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 接近零度旋转，直接返回
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
