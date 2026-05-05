"""
用预训练的 MAPPO 专家模型在多任务环境中收集数据，
保存为 buffer 可直接加载的 npz 文件。

用法:
    python -m scripts.warmup_with_demos
    python -m scripts.warmup_with_demos --config configs/mamujoco/mtmarl.yaml

原理:
    1. 创建 MultiTaskMaMuJoCo 环境（单线程）
    2. 遍历所有任务，加载对应的 MAPPO 专家模型
    3. 从统一维度的观测中提取原始维度的观测喂给专家
    4. 专家输出的动作补零到统一维度后执行
    5. 所有数据按 buffer.insert() 的格式收集并保存
"""

import argparse
import os

import numpy as np
import torch
import yaml

from baselines.models.policy.stochastic_policy import (
    StochasticPolicy,
)
from src.envs.mamujoco.multi_task import (
    MultiTaskMaMuJoCo,
)


# ================================================================
# 专家模型自动扫描
# ================================================================

# baselines 训练结果的根目录
BASELINES_RUNS_DIR = "baselines/runs"


def scan_demo_dirs(
    env: str = "mamujoco",
    algo: str = "mappo",
) -> dict[tuple[str, str], str]:
    """
    自动扫描 baselines 训练结果目录，按
    (ARGS_REGISTRY 键, 任务名) 匹配模型路径。

    扫描目录结构:
    ``{BASELINES_RUNS_DIR}/{env}/{scenario}-{conf}/{algo}/{task}/seed-.../models/``

    参数:
        env: 环境名称，如 "mamujoco"。
        algo: 算法名称，如 "mappo"。

    返回:
        { ("config_name", "task_name"): "models路径" }
    """
    from src.envs.mamujoco.tasks import ARGS_REGISTRY

    # 构建 (scenario-conf) → config_name 的反查表
    # 如 "HalfCheetah-2x3" → "2_Agent_HalfCheetah"
    conf_to_config: dict[str, str] = {}
    for config_name, args in ARGS_REGISTRY.items():
        # 从 config_name 推断 scenario
        # "2_Agent_HalfCheetah" → "HalfCheetah"
        parts = config_name.split("_", 2)
        scenario = parts[-1] if len(parts) == 3 else parts[-1]
        conf = str(args["agent_conf"]).replace(
            "|", "-",
        )
        dir_name = f"{scenario}-{conf}"
        conf_to_config[dir_name] = config_name

    demo_dirs: dict[tuple[str, str], str] = {}
    env_root = os.path.join(BASELINES_RUNS_DIR, env)
    if not os.path.isdir(env_root):
        print(
            f"  警告: 未找到目录 {env_root}"
        )
        return demo_dirs

    # 遍历 env_root 下的各 scenario 目录
    for scenario_dir in sorted(os.listdir(env_root)):
        scenario_path = os.path.join(
            env_root, scenario_dir,
        )
        if not os.path.isdir(scenario_path):
            continue

        config_name = conf_to_config.get(
            scenario_dir,
        )
        if config_name is None:
            continue

        # 在 scenario/algo/ 下查找各 task
        algo_path = os.path.join(
            scenario_path, algo,
        )
        if not os.path.isdir(algo_path):
            continue

        for task_dir in sorted(os.listdir(algo_path)):
            task_path = os.path.join(
                algo_path, task_dir,
            )
            if not os.path.isdir(task_path):
                continue

            # 找最新的 seed 目录（按名称排序取最后）
            seed_dirs = sorted([
                d for d in os.listdir(task_path)
                if d.startswith("seed-")
                and os.path.isdir(
                    os.path.join(task_path, d),
                )
            ])
            if not seed_dirs:
                continue

            models_path = os.path.join(
                task_path, seed_dirs[-1], "models",
            )
            if os.path.isdir(models_path):
                demo_dirs[
                    (config_name, task_dir)
                ] = models_path

    return demo_dirs

# 与训练时一致的 MAPPO 模型参数
MAPPO_ARGS = {
    "hidden_sizes": [128, 128],
    "activation_func": "relu",
    "use_feature_normalization": True,
    "initialization_method": "orthogonal_",
    "gain": 0.01,
    "use_naive_recurrent_policy": False,
    "use_recurrent_policy": False,
    "recurrent_n": 1,
    "use_policy_active_masks": True,
    "std_x_coef": 1,
    "std_y_coef": 0.5,
}

DEVICE = torch.device("cpu")


# ================================================================
# 辅助函数
# ================================================================

def load_demo_actors(
    env: MultiTaskMaMuJoCo,
    domain_idx: int,
    model_dir: str,
) -> list:
    """
    加载指定 domain 的 MAPPO 专家模型。

    从 checkpoint 权重推断观测维度来构建模型，
    避免与 MultiTaskMaMuJoCo 的统一维度冲突。

    参数:
        env: 多任务环境实例。
        domain_idx: domain 索引。
        model_dir: 模型文件所在目录。

    返回:
        actor 列表。
    """
    from gymnasium.spaces import Box

    agents = env._domain_agents[domain_idx]
    underlying_env = env._envs[domain_idx]

    actors = []
    for i, agent_key in enumerate(agents):
        pt_path = os.path.join(
            model_dir, f"actor_agent{i}.pt",
        )
        state_dict = torch.load(
            pt_path, map_location=DEVICE,
            weights_only=True,
        )

        # 从权重推断观测维度
        # feature_norm.weight 的 shape 就是 obs_dim
        obs_dim = state_dict[
            "base.feature_norm.weight"
        ].shape[0]
        # 动作空间从底层环境获取（维度不受统一影响）
        act_space = underlying_env.action_spaces[
            agent_key
        ]
        # 用推断的维度构建 obs_space
        obs_space = Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

        actor = StochasticPolicy(
            MAPPO_ARGS, obs_space, act_space, DEVICE,
        )
        actor.load_state_dict(state_dict)
        actor.eval()
        actors.append(actor)
    return actors


@torch.no_grad()
def demo_get_actions(
    actors: list,
    obs_n: list[np.ndarray],
    env: MultiTaskMaMuJoCo,
    domain_idx: int,
) -> list[np.ndarray]:
    """
    用专家模型从统一观测中推理动作。

    步骤:
        1. 从统一观测提取原始维度的观测（按模型
           实际输入维度截取）
        2. 喂给专家模型得到原始维度的动作
        3. 将动作补零到统一维度

    参数:
        actors: 当前 domain 的专家 actor 列表。
        obs_n: 统一维度的观测列表。
        env: 多任务环境实例。
        domain_idx: 当前 domain 索引。

    返回:
        统一维度的动作列表。
    """
    n_tasks = env._n_tasks
    n_agents = env._n_agents
    act_size = env._act_size
    act_shapes = env._domain_act_shapes[domain_idx]
    real_n_agents = len(actors)

    actions_out = []
    for i in range(n_agents):
        if i < real_n_agents:
            # 从模型的 feature_norm 推断期望的
            # 输入维度（模型训练时的观测维度）
            model_obs_dim = (
                actors[i].base.feature_norm
                .weight.shape[0]
            )
            # 提取原始观测: 跳过 one-hot 前缀
            raw_obs = obs_n[i][
                n_tasks: n_tasks + model_obs_dim
            ]
            obs_tensor = torch.FloatTensor(
                raw_obs,
            ).unsqueeze(0)

            # MAPPO 推理（不用 RNN）
            rnn_states = torch.zeros(
                1, 1,
                MAPPO_ARGS["hidden_sizes"][-1],
            )
            masks = torch.ones(1, 1)
            action, _, _ = actors[i](
                obs_tensor, rnn_states, masks,
                deterministic=True,
            )
            raw_action = action.squeeze(0).numpy()

            # 补零到统一动作维度
            padded_action = np.zeros(
                act_size, dtype=np.float32,
            )
            padded_action[:act_shapes[i]] = raw_action
            actions_out.append(padded_action)
        else:
            # 虚拟智能体，零动作
            actions_out.append(
                np.zeros(act_size, dtype=np.float32),
            )

    return actions_out


def collect_demo_data(
    env_args: dict,
    env: str = "mamujoco",
    algo: str = "mappo",
    steps_per_task: int = 2500,
    save_path: str = "demo_buffer.npz",
) -> None:
    """
    在多任务环境中用专家模型收集数据。

    参数:
        env_args: 环境配置字典。
        env: 环境名称，用于扫描模型目录。
        algo: 算法名称，用于扫描模型目录。
        steps_per_task: 每个任务收集的步数。
        save_path: 保存路径。
    """
    mt_env = MultiTaskMaMuJoCo(env_args)
    n_tasks = mt_env.n_tasks
    n_agents = mt_env.n_agents

    # 自动扫描专家模型目录
    scanned_dirs = scan_demo_dirs(
        env=env, algo=algo,
    )
    print(
        f"  扫描到 {len(scanned_dirs)} 个专家模型"
    )

    # 预加载所有 task 的专家模型
    # demo_actors[task_idx] = actor 列表
    demo_actors: dict[int, list] = {}
    config_names = list(env_args["envs"].keys())
    task_idx_counter = 0
    for domain_idx, config_name in enumerate(
        config_names
    ):
        tasks = env_args["envs"][config_name]
        for task_name in tasks:
            key = (config_name, task_name)
            if key in scanned_dirs:
                demo_actors[task_idx_counter] = (
                    load_demo_actors(
                        mt_env, domain_idx,
                        scanned_dirs[key],
                    )
                )
                print(
                    f"  已加载专家模型: "
                    f"{config_name}/{task_name}"
                )
            else:
                print(
                    f"  警告: {config_name}/{task_name}"
                    f" 无专家模型，将使用随机动作"
                )
            task_idx_counter += 1

    # 收集数据（按 buffer.insert 的格式）
    all_data = {
        "share_obs": [],
        "obs": [[] for _ in range(n_agents)],
        "actions": [[] for _ in range(n_agents)],
        "rewards": [],
        "dones": [],
        "valid_transitions": [
            [] for _ in range(n_agents)
        ],
        "terms": [],
        "next_share_obs": [],
        "next_obs": [[] for _ in range(n_agents)],
        "qpos": [],
        "qvel": [],
        "task_indices": [],
    }

    total_steps = 0
    for task_idx in range(n_tasks):
        mt_env.set_task(task_idx)
        domain_idx = mt_env._domain_indices[task_idx]
        actors = demo_actors.get(task_idx)

        obs_n, share_obs_n, _ = mt_env.reset()
        agent_deaths = np.zeros((n_agents, 1))

        for step in range(steps_per_task):
            # 记录当前 MuJoCo 物理状态
            unwrapped = (
                mt_env.env.single_agent_env.unwrapped
            )
            all_data["qpos"].append(
                unwrapped.data.qpos.copy(),
            )
            all_data["qvel"].append(
                unwrapped.data.qvel.copy(),
            )
            all_data["task_indices"].append(task_idx)

            # 获取动作
            if actors is not None:
                actions = demo_get_actions(
                    actors, obs_n, mt_env, domain_idx,
                )
            else:
                actions = [
                    mt_env.action_space[i].sample()
                    for i in range(n_agents)
                ]

            # 执行
            (
                next_obs_n, next_share_obs_n,
                reward_n, done_n, info_n, _,
            ) = mt_env.step(actions)

            # 记录有效转移
            valid_trans = 1 - agent_deaths
            agent_deaths = np.array([
                [1.0 if done_n[i] else 0.0]
                for i in range(n_agents)
            ])

            # 判断真实终止
            dones_env = all(done_n)
            term = False
            if dones_env:
                bad = (
                    "bad_transition" in info_n[0]
                    and info_n[0]["bad_transition"]
                )
                if not bad:
                    term = True

            # 存储
            all_data["share_obs"].append(
                share_obs_n[0],
            )
            for i in range(n_agents):
                all_data["obs"][i].append(obs_n[i])
                all_data["actions"][i].append(
                    actions[i],
                )
                all_data["valid_transitions"][i].append(
                    valid_trans[i],
                )
                all_data["next_obs"][i].append(
                    next_obs_n[i],
                )
            all_data["rewards"].append(
                reward_n[0],
            )
            all_data["dones"].append(
                np.array([dones_env]),
            )
            all_data["terms"].append(
                np.array([term]),
            )
            all_data["next_share_obs"].append(
                next_share_obs_n[0],
            )

            total_steps += 1

            # 回合结束则重置
            if dones_env:
                obs_n, share_obs_n, _ = mt_env.reset()
                agent_deaths = np.zeros((n_agents, 1))
            else:
                obs_n = next_obs_n
                share_obs_n = next_share_obs_n

        print(
            f"  任务 {task_idx} ({mt_env._env_names[task_idx]})"
            f" 收集完成: {steps_per_task} 步"
        )

    mt_env.close()

    # 转为 numpy 数组并保存
    save_dict = {
        "share_obs": np.array(
            all_data["share_obs"], dtype=np.float32,
        ),
        "rewards": np.array(
            all_data["rewards"], dtype=np.float32,
        ),
        "dones": np.array(all_data["dones"]),
        "terms": np.array(all_data["terms"]),
        "next_share_obs": np.array(
            all_data["next_share_obs"],
            dtype=np.float32,
        ),
    }
    for i in range(n_agents):
        save_dict[f"obs_{i}"] = np.array(
            all_data["obs"][i], dtype=np.float32,
        )
        save_dict[f"actions_{i}"] = np.array(
            all_data["actions"][i], dtype=np.float32,
        )
        save_dict[f"valid_transitions_{i}"] = np.array(
            all_data["valid_transitions"][i],
            dtype=np.float32,
        )
        save_dict[f"next_obs_{i}"] = np.array(
            all_data["next_obs"][i], dtype=np.float32,
        )

    # 渲染回放用的物理状态（变长，用 object 数组）
    save_dict["qpos"] = np.array(
        all_data["qpos"], dtype=object,
    )
    save_dict["qvel"] = np.array(
        all_data["qvel"], dtype=object,
    )
    save_dict["task_indices"] = np.array(
        all_data["task_indices"], dtype=np.int32,
    )
    save_dict["n_agents"] = n_agents
    save_dict["total_steps"] = total_steps

    np.savez_compressed(save_path, **save_dict)
    print(f"\n  专家数据已保存: {save_path}")
    print(f"  总步数: {total_steps}")
    print(
        f"  文件大小: "
        f"{os.path.getsize(save_path) / 1024 / 1024:.1f} MB"
    )


# ================================================================
# 主程序
# ================================================================

def main():
    """解析参数并收集专家数据。"""
    parser = argparse.ArgumentParser(
        description="收集多任务演示数据",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mamujoco/mtmarl.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--steps_per_task",
        type=int,
        default=2500,
        help="每个任务收集的步数",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/mamujoco/demo_buffer.npz",
        help="保存路径",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mamujoco",
        help="环境名称，用于扫描模型目录",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        help="算法名称，用于扫描模型目录",
    )
    parsed = parser.parse_args()

    with open(
        parsed.config, "r", encoding="utf-8",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    env_args = cfg["env"]

    print("=" * 60)
    print("  收集多任务演示数据")
    print(f"  扫描: {parsed.env}/{parsed.algo}")
    print("=" * 60)

    collect_demo_data(
        env_args=env_args,
        env=parsed.env,
        algo=parsed.algo,
        steps_per_task=parsed.steps_per_task,
        save_path=parsed.save_path,
    )


if __name__ == "__main__":
    main()
