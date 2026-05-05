"""World Model Runner 模块。

编排 model-based 多智能体强化学习的完整训练流程，
包括环境交互、World Model 训练（编码器/动力学/奖励）、
imagination 中的 Actor-Critic 更新、MPPI 规划和评估。
"""

from __future__ import annotations

import datetime
import itertools
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box

from src.algos.actors.world_model_actor import (
    WorldModelActor,
)
from src.algos.critics.world_model_critic import (
    WorldModelCritic,
)
from src.buffers.world_model_buffer import (
    WorldModelBuffer,
)
from src.models.dynamics import SoftMoEDynamics
from src.models.encoder import StateEncoder, encoder
from src.models.base.mlp import SimNorm
from src.models.reward import SparseMoEReward
from src.utils.config import (
    get_task_name,
    save_config,
)
from src.utils.env import (
    check,
    get_num_agents,
    make_eval_env,
    make_train_env,
    set_seed,
)
from src.utils.math import TwoHotProcessor
from src.utils.model import init_device


def _format_num(num: float) -> str:
    """将大数字格式化为带后缀的可读字符串。"""
    for suffix in ["K", "M", "G", "T"]:
        num /= 1000
        if num < 1000:
            return f"{num:.1f}{suffix}"
    return f"{num:.3f}T"


class WorldModelRunner:
    """World Model 训练运行器。

    编排以下训练流程：
        1. 环境交互收集经验，存入 replay buffer。
        2. 从 buffer 采样，联合训练 World Model
           （编码器 + 动力学 + 奖励模型）和 Critic。
        3. 在 World Model imagination 中更新 Actor。
        4. （可选）使用 MPPI 规划进行动作选择。
    """

    def __init__(
        self,
        args: dict,
        algo_args: dict,
        env_args: dict,
    ) -> None:
        """
        初始化 World Model Runner。

        参数:
            args: 命令行参数，包含 algo、env、
                exp_name 等键。
            algo_args: 算法相关参数，从配置文件加载。
            env_args: 环境相关参数，从配置文件加载。
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.state_type = env_args.get(
            "state_type", "EP",
        )
        self.policy_freq = algo_args["algo"].get(
            "policy_freq", 1,
        )
        self.fixed_order = algo_args["algo"][
            "fixed_order"
        ]

        # 规划参数
        plan_cfg = algo_args["plan"]
        self.num_pi_trajs = plan_cfg["num_pi_trajs"]
        self.num_samples = plan_cfg["num_samples"]
        self.num_elites = plan_cfg["num_elites"]
        self.plan_iter = plan_cfg["iterations"]
        self.horizon = plan_cfg["horizon"]
        self.max_std = plan_cfg["max_std"]
        self.min_std = plan_cfg["min_std"]
        self.temperature = plan_cfg["temperature"]

        # world model 参数
        wm_cfg = algo_args["world_model"]
        self.latent_dim = wm_cfg["latent_dim"]
        self.step_rho = wm_cfg["step_rho"]
        self.entropy_coef = algo_args["model"][
            "entropy_coef"
        ]

        # 初始化设备与种子
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.tpdv = dict(
            dtype=torch.float32, device=self.device,
        )

        # 初始化配置目录
        (
            self.run_dir,
            self.save_dir,
            self.log_file,
            self.task_name,
            self.expt_name,
        ) = self._init_config()

        # 创建环境
        self._init_envs()

        # 创建模型
        self._init_models()

        # 创建 actor / critic / buffer
        self._init_algo()

        # 联合优化器
        self._init_optimizer()

        # 训练状态
        self.total_it = 0

    # ===========================================================
    # 初始化辅助方法
    # ===========================================================

    def _init_envs(self) -> None:
        """创建训练和评估环境。"""
        train_cfg = self.algo_args["train"]
        n_tasks = self.env_args["n_tasks"]
        n_threads = train_cfg["n_rollout_threads"]

        num_per_task = n_threads // n_tasks
        self.task_idxes = list(
            np.arange(n_tasks).repeat(num_per_task),
        )

        self.envs = make_train_env(
            self.args["env"],
            self.algo_args["seed"]["seed"],
            n_threads,
            self.env_args,
        )

        if self.algo_args["eval"]["use_eval"]:
            self.eval_envs = make_eval_env(
                self.args["env"],
                self.algo_args["seed"]["seed"],
                self.algo_args["eval"][
                    "n_eval_rollout_threads"
                ],
                self.env_args,
            )
        else:
            self.eval_envs = None

        self.num_tasks = n_tasks
        self.num_agents = get_num_agents(
            self.args["env"], self.env_args, self.envs,
        )
        self.action_spaces = self.envs.action_space
        self.action_dims = [
            self.action_spaces[i].shape[0]
            for i in range(self.num_agents)
        ]
        self.agent_deaths = np.zeros((
            self.algo_args["train"]["n_rollout_threads"],
            self.num_agents, 1,
        ))

    def _init_models(self) -> None:
        """创建 World Model 组件。"""
        wm_cfg = self.algo_args["world_model"]

        # 各智能体的观测编码器
        self.obs_encoder: list[StateEncoder] = []
        for agent_id in range(self.num_agents):
            enc = encoder(
                obs_type="state",
                obs_dim=self.envs.observation_space[
                    agent_id
                ].shape[0],
                latent_dim=wm_cfg["latent_dim"],
                enc_dim=wm_cfg["latent_dim"],
                num_layers=wm_cfg.get(
                    "num_enc_layers", 2,
                ),
                simnorm_dim=wm_cfg.get(
                    "simnorm_dim", 8,
                ),
            )
            enc.to(self.device)
            self.obs_encoder.append(enc)

        # 集中式动力学模型
        self.dynamics_model = SoftMoEDynamics(
            latent_dim=wm_cfg["latent_dim"],
            action_dim=self.action_dims[0],
            mlp_dims=wm_cfg.get(
                "dynamics_mlp_dims", [512, 512],
            ),
            num_experts=wm_cfg["num_dynamics_experts"],
            act=SimNorm(wm_cfg.get("simnorm_dim", 8)),
            dropout=wm_cfg.get("dropout", 0.0),
        )
        self.dynamics_model.to(self.device)

        # 集中式奖励模型
        self.reward_model = SparseMoEReward(
            latent_dim=wm_cfg["latent_dim"],
            action_dim=self.action_dims[0],
            num_agents=self.num_agents,
            num_experts=wm_cfg["num_reward_experts"],
            top_k=wm_cfg.get("top_k", 2),
            num_bins=max(wm_cfg.get("num_bins", 101), 1),
            num_heads=wm_cfg.get("num_heads", 1),
            ffn_hidden=wm_cfg.get("ffn_hidden", 1024),
            head_hidden=wm_cfg.get("head_hidden", 512),
        )
        self.reward_model.to(self.device)

        # 奖励分布处理器
        self.reward_processor = TwoHotProcessor(
            num_bins=wm_cfg.get("num_bins", 101),
            vmin=wm_cfg.get("reward_min", -10.0),
            vmax=wm_cfg.get("reward_max", 10.0),
            device=self.device,
        )

    def _init_algo(self) -> None:
        """创建 Actor、Critic 和 Buffer。"""
        wm_cfg = self.algo_args["world_model"]
        latent_space = Box(
            low=-10, high=10,
            shape=(wm_cfg["latent_dim"],),
            dtype=np.float32,
        )
        joint_latent_space = Box(
            low=-10, high=10,
            shape=(
                wm_cfg["latent_dim"] * self.num_agents,
            ),
            dtype=np.float32,
        )

        algo_model_args = {
            **self.algo_args["model"],
            **self.algo_args["algo"],
        }

        # 各智能体的 Actor
        self.actor: list[WorldModelActor] = []
        for agent_id in range(self.num_agents):
            agent = WorldModelActor(
                algo_model_args,
                latent_space,
                self.action_spaces[agent_id],
                device=self.device,
            )
            self.actor.append(agent)

        # 共享的 Critic
        self.critic = WorldModelCritic(
            args={
                **self.algo_args["train"],
                **algo_model_args,
                **self.algo_args["world_model"],
            },
            share_obs_space=joint_latent_space,
            act_space=self.action_spaces,
            num_agents=self.num_agents,
            state_type=self.state_type,
            device=self.device,
        )

        # 经验回放缓冲区
        buffer_args = {
            **self.algo_args["train"],
            **algo_model_args,
            **self.env_args,
        }
        self.buffer = WorldModelBuffer(
            buffer_args,
            self.envs.share_observation_space[0],
            self.num_agents,
            self.envs.observation_space,
            self.action_spaces,
        )

        # 加载已有模型
        if self.algo_args["train"].get("model_dir"):
            self.restore()

    def _init_optimizer(self) -> None:
        """创建联合优化器。"""
        model_lr = self.algo_args["model"]["lr"]
        wm_cfg = self.algo_args["world_model"]

        param_groups = []
        for agent_id in range(self.num_agents):
            param_groups.append({
                "params": (
                    self.obs_encoder[agent_id]
                    .parameters()
                ),
                "lr": model_lr * wm_cfg.get(
                    "enc_lr_scale", 1.0,
                ),
            })
        param_groups.append({
            "params": (
                self.dynamics_model.parameters()
            ),
        })
        param_groups.append({
            "params": self.reward_model.parameters(),
        })
        param_groups.append({
            "params": itertools.chain(
                self.critic.critic.parameters(),
                self.critic.critic2.parameters(),
            ),
            "lr": model_lr,
        })
        self.model_optimizer = torch.optim.Adam(
            param_groups, lr=model_lr,
        )

    def _init_config(self) -> tuple:
        """
        初始化实验目录、保存配置并打开日志文件。

        目录层级:
        ``log_dir/env/task/algo/time-seed``

        返回:
            (run_dir, save_dir, log_file, task_name,
             expt_name) 五元组。
        """
        task_name = get_task_name(
            self.args["env"], self.env_args,
        )
        hms_time = time.strftime(
            "%Y-%m-%d-%H-%M-%S", time.localtime(),
        )
        expt_name = "-".join([
            hms_time,
            "seed-{:0>5}".format(
                self.algo_args["seed"]["seed"],
            ),
        ])
        run_dir = str(os.path.join(
            self.algo_args["logger"]["log_dir"],
            task_name,
            self.args["algo"],
            expt_name,
        ))
        os.makedirs(run_dir, exist_ok=True)

        save_config(
            self.args, self.algo_args,
            self.env_args, run_dir,
        )

        save_dir = os.path.join(run_dir, "models")
        if self.algo_args["logger"].get("save_model"):
            os.makedirs(save_dir, exist_ok=True)

        log_file = open(
            os.path.join(run_dir, "progress.txt"),
            "w", encoding="utf-8",
        )
        return (
            run_dir, save_dir, log_file,
            task_name, expt_name,
        )

    # ===========================================================
    # 主训练循环
    # ===========================================================

    def run(self) -> None:
        """执行完整的训练流程。"""
        start_time = check_time = time.time()
        train_cfg = self.algo_args["train"]
        n_threads = train_cfg["n_rollout_threads"]
        steps = (
            train_cfg["num_env_steps"] // n_threads
        )
        update_num = int(
            train_cfg["update_per_train"]
            * train_cfg["train_interval"]
        )

        episode_rewards = np.zeros(n_threads)
        done_rewards = [
            [] for _ in range(self.num_tasks)
        ]
        self.running_mean = [
            torch.zeros(
                self.horizon, n_threads,
                self.action_dims[i],
            ).to(**self.tpdv)
            for i in range(self.num_agents)
        ]
        t0 = [True] * n_threads

        # 预热
        if train_cfg.get("model_dir"):
            obs, share_obs, _ = self.envs.reset()
        else:
            obs, share_obs, _ = self.warmup()

        train_info: dict = {}

        for step in range(1, steps + 1):
            # 环境交互
            actions = self.plan(
                obs, t0=t0, add_random=True,
            )
            (
                new_obs, new_share_obs, rewards,
                dones, infos, _,
            ) = self.envs.step(actions)

            # 记录回合奖励
            dones_env = np.all(dones, axis=1)
            episode_rewards += np.mean(
                rewards, axis=1,
            ).flatten()

            for i in range(n_threads):
                if dones_env[i]:
                    done_rewards[
                        self.task_idxes[i]
                    ].append(episode_rewards[i])
                    episode_rewards[i] = 0
                    t0[i] = True
                else:
                    t0[i] = False

            # 插入 buffer
            self._insert(
                share_obs, obs, actions, rewards,
                dones, infos, new_share_obs, new_obs,
            )
            obs = new_obs
            share_obs = new_share_obs

            # 训练
            if step % train_cfg["train_interval"] == 0:
                if train_cfg.get("use_linear_lr_decay"):
                    for i in range(self.num_agents):
                        self.actor[i].lr_decay(
                            step, steps,
                        )
                    self.critic.lr_decay(step, steps)

                for _ in range(update_num):
                    info = self.train()
                    for k, v in info.items():
                        train_info.setdefault(
                            k, [],
                        ).append(v)

            # 日志
            if step % train_cfg["log_interval"] == 0:
                avg_info = {
                    k: np.mean(v)
                    for k, v in train_info.items()
                }
                rollout_info = {
                    "rew_buffer": (
                        self.buffer.get_mean_rewards()
                    ),
                }
                for tid in range(self.num_tasks):
                    if done_rewards[tid]:
                        rollout_info[f"r_{tid}"] = (
                            np.mean(done_rewards[tid])
                        )
                        done_rewards[tid] = []

                self._log(
                    step, start_time, check_time,
                    train_info=avg_info,
                    rollout_info=rollout_info,
                )
                check_time = time.time()
                train_info = {}

            # 评估
            if (step % train_cfg["eval_interval"] == 0
                    and self.algo_args["eval"][
                        "use_eval"
                    ]):
                self.eval()

            # 保存
            if (step % train_cfg["save_interval"] == 0
                    and self.algo_args["logger"].get(
                        "save_model",
                    )):
                self.save()

    # ===========================================================
    # 预热
    # ===========================================================

    def warmup(self) -> tuple:
        """
        用随机动作填充 replay buffer。

        返回:
            (obs, share_obs, available_actions) 三元组。
        """
        train_cfg = self.algo_args["train"]
        warmup_steps = (
            train_cfg["warmup_steps"]
            // train_cfg["n_rollout_threads"]
        )
        obs, share_obs, _ = self.envs.reset()

        for _ in range(warmup_steps):
            actions = self._sample_random_actions()
            (
                new_obs, new_share_obs, rewards,
                dones, infos, _,
            ) = self.envs.step(actions)
            self._insert(
                share_obs, obs, actions, rewards,
                dones, infos, new_share_obs, new_obs,
            )
            obs = new_obs
            share_obs = new_share_obs

        # 可选：预热阶段训练 world model
        wm_cfg = self.algo_args["world_model"]
        if wm_cfg.get("warmup_train"):
            for _ in range(wm_cfg.get(
                "wt_steps", 10000,
            )):
                self.train(train_actor=True)

        return obs, share_obs, None

    # ===========================================================
    # 数据插入
    # ===========================================================

    def _insert(
        self,
        share_obs: np.ndarray,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list,
        next_share_obs: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        """
        预处理交互数据并插入缓冲区。

        参数:
            share_obs: 共享观测 (n_threads, n_agents, dim)。
            obs: 观测 (n_threads, n_agents, dim)。
            actions: 动作 (n_threads, n_agents, dim)。
            rewards: 奖励 (n_threads, n_agents, 1)。
            dones: 终止标志 (n_threads, n_agents)。
            infos: 信息列表。
            next_share_obs: 下一步共享观测。
            next_obs: 下一步观测。
        """
        n_threads = self.algo_args["train"][
            "n_rollout_threads"
        ]
        dones_env = np.all(dones, axis=1)

        valid_transitions = 1 - self.agent_deaths
        self.agent_deaths = np.expand_dims(
            dones, axis=-1,
        )

        # 区分真实终止和时间截断
        terms = np.full((n_threads, 1), False)
        for i in range(n_threads):
            if dones_env[i]:
                bad = (
                    "bad_transition" in infos[i][0]
                    and infos[i][0]["bad_transition"]
                )
                if not bad:
                    terms[i][0] = True
                self.agent_deaths = np.zeros((
                    n_threads, self.num_agents, 1,
                ))

        data = (
            share_obs[:, 0],
            obs.transpose(1, 0, 2),
            actions.transpose(1, 0, 2),
            None,
            rewards[:, 0],
            np.expand_dims(dones_env, axis=-1),
            valid_transitions.transpose(1, 0, 2),
            terms,
            next_share_obs[:, 0],
            next_obs.transpose(1, 0, 2),
            None,
        )
        self.buffer.insert(data)

    # ===========================================================
    # 动作选择
    # ===========================================================

    def _sample_random_actions(self) -> np.ndarray:
        """
        从动作空间均匀采样随机动作。

        返回:
            动作数组 (n_threads, n_agents, dim)。
        """
        n_threads = self.algo_args["train"][
            "n_rollout_threads"
        ]
        actions = np.array([
            [
                self.action_spaces[i].sample()
                for i in range(self.num_agents)
            ]
            for _ in range(n_threads)
        ])
        return actions

    @torch.no_grad()
    def get_actions(
        self,
        obs: np.ndarray,
        stochastic: bool = True,
    ) -> np.ndarray:
        """
        通过策略网络选择动作。

        参数:
            obs: 观测 (n_threads, n_agents, dim)。
            stochastic: 是否使用随机策略。

        返回:
            动作数组 (n_threads, n_agents, dim)。
        """
        actions = []
        for i in range(self.num_agents):
            obs_i = check(obs[:, i]).to(**self.tpdv)
            z_i = self.obs_encoder[i](obs_i)
            a_i = self.actor[i].get_actions(
                z_i, stochastic=stochastic,
            )
            actions.append(a_i.cpu().numpy())
        return np.array(actions).transpose(1, 0, 2)

    @torch.no_grad()
    def plan(
        self,
        obs: np.ndarray,
        t0: list[bool] | None = None,
        add_random: bool = True,
    ) -> np.ndarray:
        """
        使用 MPPI 规划选择动作。

        在 World Model 中模拟多条轨迹，通过
        CEM（交叉熵方法）选出最优动作序列的
        第一步动作。

        参数:
            obs: 观测 (n_threads, n_agents, dim)。
            t0: 各线程是否为回合首步，首步时重置
                running_mean。
            add_random: 是否在选出的动作上加噪声。

        返回:
            动作数组 (n_threads, n_agents, dim)。
        """
        n_threads = obs.shape[0]
        gamma = self.algo_args["algo"]["gamma"]
        if t0 is None:
            t0 = [True] * n_threads

        # 编码观测
        zs = [
            self.obs_encoder[i](
                check(obs[:, i]).to(**self.tpdv),
            )
            for i in range(self.num_agents)
        ]

        # 初始化均值和标准差
        act_mean = [
            torch.zeros(
                self.horizon, n_threads,
                self.action_dims[i],
            ).to(**self.tpdv)
            for i in range(self.num_agents)
        ]
        act_std = [
            self.max_std * torch.ones_like(
                act_mean[i],
            )
            for i in range(self.num_agents)
        ]

        # 利用上一步的 running_mean 热启动
        for thread in range(n_threads):
            if not t0[thread]:
                for i in range(self.num_agents):
                    act_mean[i][:-1, thread] = (
                        self.running_mean[i][
                            1:, thread
                        ]
                    )

        # 采样容器
        actions = [
            torch.zeros(
                self.horizon, n_threads,
                self.num_samples, self.action_dims[i],
            ).to(**self.tpdv)
            for i in range(self.num_agents)
        ]

        # 策略轨迹：用当前策略生成部分候选
        if self.num_pi_trajs > 0:
            self._fill_pi_trajectories(
                zs, actions, n_threads,
            )

        # 输出动作
        out_a = [
            torch.zeros(
                n_threads, self.action_dims[i],
            ).to(**self.tpdv)
            for i in range(self.num_agents)
        ]

        # CEM 迭代
        for it in range(self.plan_iter):
            # 扩展 zs
            zs_exp = [
                zs[i].unsqueeze(1).repeat(
                    1, self.num_samples, 1,
                )
                for i in range(self.num_agents)
            ]

            # 高斯采样填充剩余候选
            n_random = (
                self.num_samples - self.num_pi_trajs
            )
            for i in range(self.num_agents):
                actions[i][
                    :, :, self.num_pi_trajs:
                ] = torch.normal(
                    mean=act_mean[i].unsqueeze(2)
                    .repeat(1, 1, n_random, 1),
                    std=act_std[i].unsqueeze(2)
                    .repeat(1, 1, n_random, 1),
                ).clamp(-1, 1)

            # 估计轨迹价值
            g_returns = self._estimate_value(
                zs_exp, actions, gamma,
            )

            # 对每个智能体做 CEM 更新
            for i in range(self.num_agents):
                value = torch.mean(
                    torch.stack(g_returns, dim=0),
                    dim=0,
                ).squeeze(-1)

                elite_idx = torch.topk(
                    value, self.num_elites, dim=-1,
                )[1]
                elite_val = torch.gather(
                    value, -1, elite_idx,
                )
                elite_act = torch.gather(
                    actions[i], 2,
                    elite_idx.unsqueeze(0).unsqueeze(-1)
                    .repeat(
                        self.horizon, 1, 1,
                        self.action_dims[i],
                    ),
                )

                # 加权得分
                max_v = elite_val.max(
                    dim=-1, keepdim=True,
                )[0]
                score = torch.exp(
                    self.temperature
                    * (elite_val - max_v)
                )
                score = score / score.sum(
                    dim=-1, keepdim=True,
                )
                w = score.unsqueeze(0).unsqueeze(-1)

                act_mean[i] = (
                    w * elite_act
                ).sum(dim=2)
                act_std[i] = torch.sqrt(
                    (
                        w
                        * (
                            elite_act
                            - act_mean[i].unsqueeze(2)
                        ) ** 2
                    ).sum(dim=2)
                    + 1e-6
                ).clamp_(self.min_std, self.max_std)

                # 最后一次迭代选出动作
                if it == self.plan_iter - 1:
                    s = score.squeeze(0).squeeze(
                        -1,
                    ).cpu().numpy()
                    for t in range(n_threads):
                        idx = np.random.choice(
                            self.num_elites, p=s[t],
                        )
                        out_a[i][t] = (
                            elite_act[0, t, idx]
                        )
                        if add_random:
                            out_a[i][t] += (
                                torch.randn_like(
                                    out_a[i][t],
                                )
                                * act_std[i][0, t]
                            )
                            out_a[i][t].clamp_(-1, 1)

        self.running_mean = act_mean
        result = np.array([
            out_a[i].cpu().numpy()
            for i in range(self.num_agents)
        ]).transpose(1, 0, 2)
        return result

    def _fill_pi_trajectories(
        self,
        zs: list[torch.Tensor],
        actions: list[torch.Tensor],
        n_threads: int,
    ) -> None:
        """
        用当前策略在 World Model 中 rollout，填充
        策略轨迹候选到 actions 的前 num_pi_trajs 列。

        参数:
            zs: 各智能体的 latent 状态列表。
            actions: 采样容器，原地修改。
            n_threads: 并行环境数量。
        """
        pi_act = [
            torch.zeros(
                self.horizon, n_threads,
                self.num_pi_trajs,
                self.action_dims[i],
            ).to(**self.tpdv)
            for i in range(self.num_agents)
        ]
        cur_zs = torch.stack([
            zs[i].unsqueeze(1).repeat(
                1, self.num_pi_trajs, 1,
            )
            for i in range(self.num_agents)
        ], dim=0)

        for t in range(self.horizon - 1):
            for i in range(self.num_agents):
                pi_act[i][t] = self.actor[
                    i
                ].get_actions(
                    cur_zs[i], stochastic=True,
                )

            flat_z = cur_zs.permute(
                1, 2, 0, 3,
            ).reshape(
                n_threads * self.num_pi_trajs,
                self.num_agents, -1,
            )
            flat_a = torch.stack([
                pi_act[i][t]
                for i in range(self.num_agents)
            ], dim=-2).reshape(
                n_threads * self.num_pi_trajs,
                self.num_agents, -1,
            )
            next_z = self.dynamics_model(
                flat_z, flat_a,
            ).reshape(
                n_threads, self.num_pi_trajs,
                self.num_agents, -1,
            )
            cur_zs = next_z.permute(
                2, 0, 1, 3,
            ).contiguous()

        # 最后一步
        for i in range(self.num_agents):
            pi_act[i][-1] = self.actor[
                i
            ].get_actions(
                cur_zs[i], stochastic=True,
            )
            actions[i][
                :, :, :self.num_pi_trajs, :
            ] = pi_act[i].clone()

    @torch.no_grad()
    def _estimate_value(
        self,
        zs: list[torch.Tensor],
        actions: list[torch.Tensor],
        gamma: float,
    ) -> list[torch.Tensor]:
        """
        在 World Model 中估计轨迹价值。

        参数:
            zs: 各智能体 latent 状态，每个形状为
                (n_threads, num_samples, dim)。
            actions: 各智能体动作序列，每个形状为
                (horizon, n_threads, num_samples, dim)。
            gamma: 折扣因子。

        返回:
            各智能体的回报列表，每个形状为
            (n_threads, num_samples, 1)。
        """
        horizon = actions[0].shape[0]
        n_threads = actions[0].shape[1]
        num_samples = actions[0].shape[2]

        cur_zs = torch.stack(
            zs, dim=-2,
        )  # (nt, ns, na, d)
        actions_t = torch.stack(
            actions, dim=-2,
        )  # (h, nt, ns, na, d)

        returns = [
            torch.zeros(
                horizon + 1, n_threads,
                num_samples, 1,
            ).to(**self.tpdv)
            for _ in range(self.num_agents)
        ]

        for t in range(horizon):
            flat_z = cur_zs.reshape(
                n_threads * num_samples,
                self.num_agents, -1,
            )
            flat_a = actions_t[t].reshape(
                n_threads * num_samples,
                self.num_agents, -1,
            )

            z_pred = self.dynamics_model(
                flat_z, flat_a,
            ).reshape(*cur_zs.shape)

            r_logits, _ = self.reward_model(
                flat_z, flat_a,
            )
            r_val = self.reward_processor.decode(
                r_logits.reshape(
                    n_threads, num_samples, -1,
                ),
            )

            for i in range(self.num_agents):
                returns[i][t + 1] = (
                    returns[i][t]
                    + gamma ** t * r_val
                )
            cur_zs = z_pred

        # 终端 Q 值
        joint_z = cur_zs.reshape(
            n_threads, num_samples, -1,
        )
        joint_a = torch.cat([
            self.actor[i].get_actions(
                cur_zs[:, :, i], stochastic=True,
            )
            for i in range(self.num_agents)
        ], dim=-1)
        terminal_q = self.critic.get_values(
            joint_z, joint_a, mode="mean",
        )

        g_returns = []
        for i in range(self.num_agents):
            final = (
                returns[i][-2]
                + gamma ** horizon * terminal_q
            )
            g_returns.append(final.nan_to_num(0))

        return g_returns

    # ===========================================================
    # 训练
    # ===========================================================

    def train(
        self,
        train_actor: bool = True,
    ) -> dict:
        """
        执行一次训练迭代。

        参数:
            train_actor: 是否同时训练 Actor。

        返回:
            训练信息字典。
        """
        if self.buffer.cur_size < self.buffer.batch_size:
            return {}
        self.total_it += 1

        # 采样
        t0 = torch.randperm(
            self.buffer.cur_size,
        ).numpy()[:self.buffer.batch_size]
        self.buffer._update_end_flag()
        indices = [t0]
        for _ in range(self.horizon - 1):
            indices.append(
                self.buffer._next_indices(
                    indices[-1],
                ),
            )

        data_h = self.buffer.sample_horizon(
            horizon=self.horizon, t0=t0,
        )
        (
            _, sp_obs, sp_actions, _, sp_reward,
            _, _, sp_term, _, sp_next_obs, _, _,
        ) = data_h

        # 对每个时间步计算 n-step 目标
        sp_nstep_reward = np.zeros_like(sp_reward)
        sp_nstep_term = np.zeros_like(sp_term)
        sp_nstep_next_obs = np.zeros_like(sp_next_obs)
        sp_nstep_gamma = np.zeros_like(sp_reward)
        for t, idx in enumerate(indices):
            data_n = self.buffer.sample(indices=idx)
            (
                _, _, _, _, nr, _, _, nt,
                _, nno, _, ng, _, _,
            ) = data_n
            sp_nstep_reward[t] = nr
            sp_nstep_term[t] = nt
            sp_nstep_next_obs[:, t] = nno
            sp_nstep_gamma[t] = ng

        # 训练 World Model + Critic
        train_info, zs = self._model_train(
            sp_obs, sp_actions, sp_reward,
            sp_next_obs, sp_nstep_reward,
            sp_nstep_term, sp_nstep_next_obs,
            sp_nstep_gamma,
        )

        # 训练 Actor
        if (self.total_it % self.policy_freq == 0
                and train_actor):
            actor_info = self._actor_train(
                zs=[
                    zs[i, :-1]
                    for i in range(self.num_agents)
                ],
            )
            train_info.update(actor_info)

        self.critic.soft_update()
        return train_info

    def _model_train(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        nstep_reward: np.ndarray,
        nstep_term: np.ndarray,
        nstep_next_obs: np.ndarray,
        nstep_gamma: np.ndarray,
    ) -> tuple[dict, torch.Tensor]:
        """
        训练 World Model（编码器、动力学、奖励）和
        Critic 的 Q 网络。

        参数:
            obs: (n_agents, horizon, batch, dim)。
            actions: (n_agents, horizon, batch, dim)。
            reward: (horizon, batch, 1)。
            next_obs: (n_agents, horizon, batch, dim)。
            nstep_reward: (horizon, batch, 1)。
            nstep_term: (horizon, batch, 1)。
            nstep_next_obs: (n_agents, horizon, batch, dim)。
            nstep_gamma: (horizon, batch, 1)。

        返回:
            (train_info, zs) 元组，其中 zs 形状为
            (n_agents, horizon+1, batch, latent_dim)。
        """
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        reward = check(reward).to(**self.tpdv)
        next_obs = check(next_obs).to(**self.tpdv)
        nstep_reward = check(
            nstep_reward,
        ).to(**self.tpdv)
        nstep_term = check(
            nstep_term,
        ).to(**self.tpdv)
        nstep_next_obs = check(
            nstep_next_obs,
        ).to(**self.tpdv)
        nstep_gamma = check(
            nstep_gamma,
        ).to(**self.tpdv)

        n_agents, horizon, batch_size, _ = obs.shape

        # 计算目标 Q 值
        with torch.no_grad():
            next_zs = [
                self.obs_encoder[i](next_obs[i])
                for i in range(self.num_agents)
            ]
            nstep_zs = [
                self.obs_encoder[i](nstep_next_obs[i])
                for i in range(self.num_agents)
            ]
            nstep_acts = [
                self.actor[i].get_actions(nstep_zs[i])
                for i in range(self.num_agents)
            ]
            next_zs = torch.stack(next_zs, dim=0)

        next_q = self.critic.get_target_values(
            torch.cat(nstep_zs, dim=-1),
            torch.cat(nstep_acts, dim=-1),
        )
        q_targets = (
            nstep_reward
            + nstep_gamma * next_q * (1 - nstep_term)
        )

        # 开启梯度
        self._model_turn_on_grad()

        # h-step 预测损失
        dyn_loss = 0.0
        rew_loss = 0.0
        q_loss = 0.0
        bal_loss = 0.0

        zs = torch.zeros(
            self.num_agents, horizon + 1,
            batch_size, self.latent_dim,
        ).to(**self.tpdv)
        for i in range(self.num_agents):
            zs[i, 0] = self.obs_encoder[i](
                obs[i, 0],
            )

        train_info: dict = {
            "reward_acc": 0.0, "reward_err": 0.0,
        }
        rho = self.step_rho

        for t in range(horizon):
            # (batch, n_agents, dim)
            z_in = zs[:, t].permute(1, 0, 2)
            a_in = actions[:, t].permute(1, 0, 2)

            z_pred = self.dynamics_model(
                z_in, a_in,
            ).permute(1, 0, 2)

            r_logits, aux = self.reward_model(
                z_in, a_in,
            )

            dyn_loss += F.mse_loss(
                z_pred, next_zs[:, t],
            ).mean() * (rho ** t)

            rew_loss += self.reward_processor.loss(
                r_logits, reward[t],
            ).mean() * (rho ** t)

            bal_loss += (
                aux["balance_loss"] * (rho ** t)
            )

            zs[:, t + 1] = z_pred.clone()

            # Q 损失
            joint_z = torch.cat([
                zs[i, t]
                for i in range(self.num_agents)
            ], dim=-1)
            joint_a = torch.cat([
                actions[i, t]
                for i in range(self.num_agents)
            ], dim=-1)
            q1 = self.critic.critic(joint_z, joint_a)
            q2 = self.critic.critic2(joint_z, joint_a)
            q_loss += (
                (
                    self.critic.processor.loss(
                        q1, q_targets[t],
                    ).mean()
                    + self.critic.processor.loss(
                        q2, q_targets[t],
                    ).mean()
                )
                / 2 * (rho ** t)
            )

            # 监控指标
            with torch.no_grad():
                r_decoded = self.reward_processor.decode(
                    r_logits,
                )
                err = torch.abs(reward[t] - r_decoded)
                train_info["reward_acc"] += (
                    (err <= 0.05).sum().item()
                    / err.shape[0] / horizon
                )
                train_info["reward_err"] += (
                    err.mean().item()
                    / self.num_agents / horizon
                )

        dyn_loss /= horizon
        rew_loss /= horizon
        q_loss /= horizon
        bal_loss /= horizon

        # 加权求和
        wm_cfg = self.algo_args["world_model"]
        total_loss = (
            q_loss * wm_cfg.get("q_coef", 1.0)
            + rew_loss * wm_cfg.get("reward_coef", 1.0)
            + dyn_loss * wm_cfg.get(
                "dynamics_coef", 1.0,
            )
            + bal_loss * wm_cfg.get(
                "balance_coef", 0.01,
            )
        )

        self.model_optimizer.zero_grad()
        total_loss.backward()
        for group in self.model_optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(
                group["params"], 20,
            )
        self.model_optimizer.step()

        self._model_turn_off_grad()

        train_info.update({
            "dynamics_loss": float(dyn_loss),
            "reward_loss": float(rew_loss),
            "q_loss": float(q_loss),
            "balance_loss": float(bal_loss),
            "total_loss": float(total_loss),
        })
        return train_info, zs.detach()

    def _actor_train(
        self,
        zs: list[torch.Tensor],
    ) -> dict:
        """
        在 imagination latent 上训练 Actor。

        参数:
            zs: 各智能体的 latent 序列列表，
                每个形状为 (horizon, batch, dim)。

        返回:
            训练信息字典。
        """
        train_info: dict = {
            "actor_loss": [0.0] * self.num_agents,
        }

        # 先用无梯度策略填充所有智能体的动作
        actions = []
        logp_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                a, lp = self.actor[
                    i
                ].get_actions_with_logprobs(zs[i])
                actions.append(a)
                logp_actions.append(lp)

        # 顺序或随机序更新各智能体
        if self.fixed_order:
            order = list(range(self.num_agents))
        else:
            order = list(
                np.random.permutation(self.num_agents),
            )

        rho = self.step_rho

        for agent_id in order:
            self.actor[agent_id].turn_on_grad()

            actions[agent_id], logp_actions[agent_id] = (
                self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    zs[agent_id],
                )
            )

            value_pred = self.critic.get_values(
                torch.cat(zs, dim=-1),
                torch.cat(actions, dim=-1),
                mode="mean",
            )

            # 更新 Q 值缩放
            self.critic.scale.update(value_pred[0])
            value_pred = self.critic.scale(value_pred)

            # 带时间衰减的策略损失
            actor_loss = torch.zeros(1).to(**self.tpdv)
            for t in range(len(zs[agent_id])):
                actor_loss += (
                    self.entropy_coef
                    * logp_actions[agent_id][t]
                    - value_pred[t]
                ).mean() * (rho ** t)
            actor_loss /= len(zs[agent_id])

            self.actor[
                agent_id
            ].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[
                agent_id
            ].actor_optimizer.step()
            self.actor[agent_id].turn_off_grad()

            # 用更新后的策略刷新动作
            actions[agent_id], _ = self.actor[
                agent_id
            ].get_actions_with_logprobs(zs[agent_id])

            train_info["actor_loss"][agent_id] = (
                actor_loss.item()
            )
            train_info["pi_scale"] = (
                self.critic.scale.value
            )

        return train_info

    # ===========================================================
    # 梯度控制
    # ===========================================================

    def _model_turn_on_grad(self) -> None:
        """开启所有 World Model 和 Critic 的梯度。"""
        for i in range(self.num_agents):
            for p in self.obs_encoder[i].parameters():
                p.requires_grad = True
        for p in self.dynamics_model.parameters():
            p.requires_grad = True
        for p in self.reward_model.parameters():
            p.requires_grad = True
        self.critic.turn_on_grad()

    def _model_turn_off_grad(self) -> None:
        """关闭所有 World Model 和 Critic 的梯度。"""
        for i in range(self.num_agents):
            for p in self.obs_encoder[i].parameters():
                p.requires_grad = False
        for p in self.dynamics_model.parameters():
            p.requires_grad = False
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.critic.turn_off_grad()

    # ===========================================================
    # 评估
    # ===========================================================

    @torch.no_grad()
    def eval(self) -> None:
        """在评估环境上运行并输出结果。"""
        if self.eval_envs is None:
            return

        eval_cfg = self.algo_args["eval"]
        n_threads = eval_cfg["n_eval_rollout_threads"]
        target_episodes = eval_cfg["eval_episodes"]
        use_plan = self.algo_args["plan"].get(
            "use_plan", True,
        )

        episode_rewards: list[list] = [
            [] for _ in range(n_threads)
        ]
        one_ep_rewards: list[list] = [
            [] for _ in range(n_threads)
        ]
        episode_lens: list[int] = []
        one_ep_len = np.zeros(n_threads, dtype=np.int32)
        done_count = 0
        t0 = [True] * n_threads

        obs, _, _ = self.eval_envs.reset()

        while done_count < target_episodes:
            if use_plan:
                actions = self.plan(
                    obs, t0=t0, add_random=False,
                )
            else:
                actions = self.get_actions(
                    obs, stochastic=False,
                )

            (
                new_obs, _, rewards,
                dones, _, _,
            ) = self.eval_envs.step(actions)

            obs = new_obs
            dones_env = np.all(dones, axis=1)
            one_ep_len += 1

            for i in range(n_threads):
                one_ep_rewards[i].append(rewards[i])
                if dones_env[i]:
                    done_count += 1
                    episode_rewards[i].append(
                        np.sum(
                            one_ep_rewards[i], axis=0,
                        ),
                    )
                    one_ep_rewards[i] = []
                    episode_lens.append(
                        one_ep_len[i],
                    )
                    one_ep_len[i] = 0
                    t0[i] = True
                else:
                    t0[i] = False

        all_rewards = np.concatenate([
            r for r in episode_rewards if r
        ])
        print(
            f"  评估: 平均奖励={np.mean(all_rewards):.2f}"
            f"  平均步数={np.mean(episode_lens):.0f}"
        )

    # ===========================================================
    # 持久化
    # ===========================================================

    def save(self) -> None:
        """保存所有模型参数。"""
        for i in range(self.num_agents):
            self.actor[i].save(self.save_dir, i)
            torch.save(
                self.obs_encoder[i].state_dict(),
                os.path.join(
                    self.save_dir,
                    f"obs_encoder_{i}.pt",
                ),
            )
        self.critic.save(self.save_dir)
        torch.save(
            self.dynamics_model.state_dict(),
            os.path.join(
                self.save_dir, "dynamics_model.pt",
            ),
        )
        torch.save(
            self.reward_model.state_dict(),
            os.path.join(
                self.save_dir, "reward_model.pt",
            ),
        )

    def restore(self) -> None:
        """加载所有模型参数。"""
        model_dir = self.algo_args["train"][
            "model_dir"
        ]
        for i in range(self.num_agents):
            self.actor[i].restore(model_dir, i)
            self.obs_encoder[i].load_state_dict(
                torch.load(
                    os.path.join(
                        model_dir,
                        f"obs_encoder_{i}.pt",
                    ),
                    map_location=self.device,
                ),
            )
        self.critic.restore(model_dir)
        self.dynamics_model.load_state_dict(
            torch.load(
                os.path.join(
                    model_dir, "dynamics_model.pt",
                ),
                map_location=self.device,
            ),
        )
        self.reward_model.load_state_dict(
            torch.load(
                os.path.join(
                    model_dir, "reward_model.pt",
                ),
                map_location=self.device,
            ),
        )

    def close(self) -> None:
        """关闭环境和日志文件。"""
        self.envs.close()
        if (self.eval_envs is not None
                and self.eval_envs is not self.envs):
            self.eval_envs.close()
        self.log_file.close()

    # ===========================================================
    # 日志
    # ===========================================================

    def _log(
        self,
        step: int,
        start_time: float,
        check_time: float,
        **kwargs: dict,
    ) -> None:
        """
        输出训练日志到控制台和日志文件。

        参数:
            step: 当前环境步数。
            start_time: 训练开始时间戳。
            check_time: 上次日志时间戳。
            **kwargs: 各类信息字典。
        """
        n_threads = self.algo_args["train"][
            "n_rollout_threads"
        ]
        total_steps = self.algo_args["train"][
            "num_env_steps"
        ]
        elapsed = datetime.timedelta(
            seconds=int(time.time() - start_time),
        )
        iter_time = time.time() - check_time

        lines = [
            "",
            f"******** "
            f"iter: {_format_num(self.total_it)}, "
            f"steps: "
            f"{_format_num(step * n_threads)}"
            f"/{_format_num(total_steps)}, "
            f"iter_time: {iter_time:.1f}s, "
            f"total: {elapsed} ********",
        ]

        for key, value in kwargs.items():
            parts = ", ".join(
                f"{k}: {v:.6f}"
                for k, v in value.items()
            )
            lines.append(f"{key}: {parts}")

        for line in lines:
            print(line)
            self.log_file.write(line + "\n")
        self.log_file.flush()
