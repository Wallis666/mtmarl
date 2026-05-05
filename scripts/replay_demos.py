"""
从 demo_buffer.npz 中加载 qpos/qvel，精确回放
机器人运动并渲染。

用法:
    python -m scripts.replay_demo_buffer
    python -m scripts.replay_demo_buffer --task_idx 0 --save_video cheetah.mp4
    python -m scripts.replay_demo_buffer --task_idx 2 --render_mode human
"""

import argparse
import os
import time

import numpy as np
import yaml

from src.envs.mamujoco.tasks import (
    ARGS_REGISTRY,
    ENV_REGISTRY,
)


def get_domain_info(env_args: dict) -> list[dict]:
    """
    从配置中提取各 domain 信息。

    返回:
        每个 domain 的配置列表，包含 config_name、
        scenario、tasks 等。
    """
    domains = []
    for config_name, tasks in env_args["envs"].items():
        args = ARGS_REGISTRY[config_name]
        # 从 config_name 推断 scenario
        # 如 "2_Agent_HalfCheetah" -> "HalfCheetah"
        scenario = config_name.split("_", 2)[-1]
        domains.append({
            "config_name": config_name,
            "scenario": scenario,
            "agent_conf": args["agent_conf"],
            "agent_obsk": args["agent_obsk"],
            "tasks": tasks,
        })
    return domains


def replay(
    config_path: str,
    npz_path: str,
    task_idx: int = 0,
    max_steps: int = 500,
    render_mode: str = "human",
    save_video: str = "",
) -> None:
    """
    从 npz 加载 qpos/qvel 精确回放渲染。

    参数:
        config_path: 配置文件路径。
        npz_path: 专家数据文件路径。
        task_idx: 要回放的全局任务索引。
        max_steps: 最大回放步数。
        render_mode: 渲染模式。
        save_video: 视频保存路径。
    """
    with open(
        config_path, "r", encoding="utf-8",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    env_args = cfg["env"]

    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    task_indices = data["task_indices"]
    qpos_all = data["qpos"]
    qvel_all = data["qvel"]
    rewards = data["rewards"]
    dones = data["dones"]

    # 筛选指定任务的步
    mask = task_indices == task_idx
    indices = np.where(mask)[0]
    if len(indices) == 0:
        print(f"  错误: 没有任务 {task_idx} 的数据")
        return

    indices = indices[:max_steps]
    actual_steps = len(indices)
    print(f"  任务索引: {task_idx}")
    print(f"  回放步数: {actual_steps}")

    # 确定该任务对应的 domain 和 scenario
    domains = get_domain_info(env_args)
    # 从展平的 task 列表中找到对应 domain
    task_count = 0
    target_domain = None
    target_task = None
    for domain in domains:
        for task_name in domain["tasks"]:
            if task_count == task_idx:
                target_domain = domain
                target_task = task_name
            task_count += 1

    if target_domain is None:
        print(f"  错误: 任务索引 {task_idx} 超出范围")
        return

    print(
        f"  环境: {target_domain['scenario']}"
        f"  任务: {target_task}"
    )

    # 创建对应的底层环境（带渲染）
    actual_render = (
        "rgb_array" if save_video else render_mode
    )
    scenario = target_domain["scenario"]
    env_cls = ENV_REGISTRY[scenario]
    env = env_cls(
        agent_conf=target_domain["agent_conf"],
        agent_obsk=target_domain["agent_obsk"],
        render_mode=actual_render,
    )
    env.set_task(target_task)
    env.reset(seed=42)

    unwrapped = env.single_agent_env.unwrapped

    frames = []
    total_reward = 0.0

    for i, idx in enumerate(indices):
        # 用 qpos/qvel 精确恢复物理状态
        qpos = qpos_all[idx]
        qvel = qvel_all[idx]
        unwrapped.set_state(qpos, qvel)

        # 渲染
        frame = env.render()
        if save_video and frame is not None:
            frames.append(frame)
        elif render_mode == "human":
            time.sleep(0.02)

        total_reward += rewards[idx][0]

        # 回合结束标记
        if dones[idx][0]:
            print(
                f"  回合在第 {i + 1} 步结束，"
                f"累计奖励: {total_reward:.2f}"
            )
            total_reward = 0.0

    env.close()
    if total_reward > 0:
        print(
            f"  回放完成，最终累计奖励: "
            f"{total_reward:.2f}"
        )
    else:
        print("  回放完成")

    # 保存视频
    if save_video and frames:
        try:
            import imageio
            imageio.mimsave(
                save_video, frames, fps=50,
            )
            print(
                f"  视频已保存: {save_video} "
                f"({len(frames)} 帧, "
                f"{frames[0].shape})"
            )
        except ImportError:
            print(
                "  需要 imageio 库: "
                "pip install imageio[ffmpeg]"
            )


def main():
    """解析参数并回放。"""
    parser = argparse.ArgumentParser(
        description="从专家数据精确回放渲染",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mamujoco/mtmarl.yaml",
    )
    parser.add_argument(
        "--npz",
        type=str,
        default="data/mamujoco/demo_buffer.npz",
    )
    parser.add_argument(
        "--task_idx",
        type=int,
        default=-1,
        help="全局任务索引，-1 表示所有任务",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default="",
    )
    parsed = parser.parse_args()

    print("=" * 60)
    print("  回放专家数据（qpos/qvel 精确恢复）")
    print("=" * 60)

    if parsed.task_idx >= 0:
        # 单任务回放
        replay(
            config_path=parsed.config,
            npz_path=parsed.npz,
            task_idx=parsed.task_idx,
            max_steps=parsed.max_steps,
            render_mode=parsed.render_mode,
            save_video=parsed.save_video,
        )
    else:
        # 所有任务依次回放
        with open(
            parsed.config, "r", encoding="utf-8",
        ) as f:
            cfg = yaml.load(
                f, Loader=yaml.FullLoader,
            )
        env_args = cfg["env"]
        n_tasks = env_args["n_tasks"]

        for tid in range(n_tasks):
            # 生成每个任务的视频文件名
            video_path = ""
            if parsed.save_video:
                base, ext = os.path.splitext(
                    parsed.save_video,
                )
                video_path = f"{base}_task{tid}{ext}"

            print(f"\n--- 任务 {tid}/{n_tasks} ---")
            replay(
                config_path=parsed.config,
                npz_path=parsed.npz,
                task_idx=tid,
                max_steps=parsed.max_steps,
                render_mode=parsed.render_mode,
                save_video=video_path,
            )


if __name__ == "__main__":
    main()
