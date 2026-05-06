"""
渲染训练好的 MTMARL 模型。

直接创建 WorldModelRunner 实例并加载模型，
用 runner 自身的 plan() 或 get_actions() 渲染。

用法:
    # 用 MPPI 规划渲染所有任务
    python -m scripts.render_mtmarl --use_plan \
        --model_dir runs/mamujoco/mtmarl/.../models

    # 纯 actor 推理
    python -m scripts.render_mtmarl \
        --model_dir runs/mamujoco/mtmarl/.../models

    # 指定任务 + 录制视频
    python -m scripts.render_mtmarl --use_plan --task_idx 0 \
        --model_dir runs/mamujoco/mtmarl/.../models \
        --save_video runs/videos/mtmarl
"""

import argparse
import os
import sys
import time

if sys.platform == "linux" and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import yaml

from src.envs.mamujoco.multi_task import (
    MultiTaskMaMuJoCo,
)
from src.runners.world_model_runner import (
    WorldModelRunner,
)
from src.utils.env import check


def render_task(
    runner: WorldModelRunner,
    task_idx: int,
    use_plan: bool = True,
    render_mode: str = "human",
    max_steps: int = 1000,
    save_video: str = "",
) -> float:
    """
    用 runner 的完整 plan/get_actions 渲染一个任务。

    返回:
        回合总奖励。
    """
    env_args = runner.env_args
    num_agents = runner.num_agents
    tpdv = runner.tpdv

    # 创建带渲染的单环境
    actual_render = (
        "rgb_array" if save_video else render_mode
    )
    mt_env = MultiTaskMaMuJoCo(
        env_args, render_mode=actual_render,
    )
    mt_env.set_task(task_idx)
    obs_n, _, _ = mt_env.reset(seed=42)

    # 转成 runner 期望的 (1, n_agents, dim) 格式
    obs = np.array(obs_n)[np.newaxis, :, :]

    frames = []
    total_reward = 0.0
    t0 = [True]

    for step in range(max_steps):
        if use_plan:
            actions = runner.plan(
                obs, t0=t0, add_random=False,
            )
        else:
            actions = runner.get_actions(
                obs, stochastic=False,
            )

        # actions: (1, n_agents, dim) → 列表
        actions_list = [
            actions[0, i] for i in range(num_agents)
        ]

        # World Model 预测奖励 + 门控
        with torch.no_grad():
            zs_diag = []
            acts_diag = []
            for i in range(num_agents):
                obs_t = check(
                    np.array(
                        obs_n[i], dtype=np.float32,
                    ),
                ).unsqueeze(0).to(**tpdv)
                zs_diag.append(
                    runner.obs_encoder[i](obs_t),
                )
                acts_diag.append(check(
                    np.array(
                        actions_list[i],
                        dtype=np.float32,
                    ),
                ).unsqueeze(0).to(**tpdv))

            z_batch = torch.stack(zs_diag, dim=1)
            a_batch = torch.stack(acts_diag, dim=1)
            r_logits, aux = runner.reward_model(
                z_batch, a_batch,
            )
            wm_reward = runner.reward_processor.decode(
                r_logits,
            ).item()

            gates = aux["gates"]
            gate_str = " ".join([
                f"e{j}={gates[0, j]:.2f}"
                for j in range(gates.shape[1])
            ])

        # 环境 step
        (
            obs_n, _, reward_n,
            done_n, _, _,
        ) = mt_env.step(actions_list)

        real_reward = reward_n[0][0]
        total_reward += real_reward
        t0 = [False]

        # 更新 obs
        obs = np.array(obs_n)[np.newaxis, :, :]

        # 每 50 步打印诊断
        if step % 50 == 0:
            print(
                f"  step={step:4d}  "
                f"real_r={real_reward:.4f}  "
                f"wm_r={wm_reward:.4f}  "
                f"gates=[{gate_str}]"
            )

        # 渲染
        frame = mt_env.render()
        if save_video and frame is not None:
            frames.append(frame)
        elif render_mode == "human":
            time.sleep(0.02)

        if all(done_n):
            t0 = [True]
            break

    mt_env.close()

    # 保存视频
    if save_video and frames:
        try:
            import imageio
            os.makedirs(
                os.path.dirname(save_video)
                or ".",
                exist_ok=True,
            )
            imageio.mimsave(
                save_video, frames, fps=50,
            )
            print(
                f"  视频已保存: {save_video} "
                f"({len(frames)} 帧)"
            )
        except ImportError:
            print(
                "  需要 imageio: "
                "pip install imageio[ffmpeg]"
            )

    return total_reward


def main():
    """解析参数并渲染。"""
    parser = argparse.ArgumentParser(
        description="渲染训练好的 MTMARL 模型",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mamujoco/mtmarl.yaml",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="模型目录路径",
    )
    parser.add_argument(
        "--task_idx",
        type=int,
        default=-1,
        help="任务索引，-1 表示所有任务",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default="",
        help="视频保存目录，为空则不保存",
    )
    parser.add_argument(
        "--use_plan",
        action="store_true",
        help="使用 MPPI 规划而非纯 actor 推理",
    )
    parsed = parser.parse_args()

    # 加载配置
    with open(
        parsed.config, "r", encoding="utf-8",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    env_args = cfg.pop("env")
    algo_args = cfg

    # 强制设置：不训练，不评估，加载模型
    algo_args["train"]["model_dir"] = (
        parsed.model_dir
    )
    algo_args["eval"]["use_eval"] = False
    algo_args["logger"]["save_model"] = False
    # 用 1 个线程即可
    algo_args["train"]["n_rollout_threads"] = 1

    args = {
        "algo": "mtmarl",
        "env": "mamujoco",
        "exp_name": "render",
    }

    mode_str = (
        "MPPI 规划" if parsed.use_plan
        else "Actor 推理"
    )
    print("=" * 60)
    print(f"  渲染 MTMARL 模型（{mode_str}）")
    print(f"  模型: {parsed.model_dir}")
    print("=" * 60)

    # 创建 runner 并加载模型
    runner = WorldModelRunner(
        args, algo_args, env_args,
    )

    n_tasks = env_args["n_tasks"]
    task_names = []
    for config_name, tasks in env_args["envs"].items():
        for task in tasks:
            task_names.append(
                f"{config_name}_{task}",
            )

    if parsed.task_idx >= 0:
        task_list = [parsed.task_idx]
    else:
        task_list = list(range(n_tasks))

    for tid in task_list:
        print(
            f"\n--- 任务 {tid}: {task_names[tid]} ---"
        )

        video_path = ""
        if parsed.save_video:
            video_path = os.path.join(
                parsed.save_video,
                f"{task_names[tid]}.mp4",
            )

        reward = render_task(
            runner=runner,
            task_idx=tid,
            use_plan=parsed.use_plan,
            render_mode=parsed.render_mode,
            max_steps=parsed.max_steps,
            save_video=video_path,
        )
        print(f"  回合奖励: {reward:.2f}")

    runner.close()


if __name__ == "__main__":
    main()
