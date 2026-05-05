"""
MaMuJoCo 多任务统一封装模块。

将多个异构的 MaMuJoCo 环境封装为统一接口，
使上层算法可以在多个任务之间无缝切换训练和评估。

核心功能:
    - 观测对齐: one-hot 任务编码 + 零填充至统一维度
    - 动作对齐: 统一动作空间 + 动作掩码
    - 智能体对齐: 补充虚拟智能体至最大数量
    - 任务切换: 通过全局索引在不同 domain/task 间切换
"""

from typing import Any

import numpy as np
from gymnasium.spaces import Box
from numpy.typing import NDArray

from src.envs.mamujoco.tasks import ARGS_REGISTRY, ENV_REGISTRY


class MultiTaskMaMuJoCo:
    """
    MaMuJoCo 多任务统一封装器。

    将多个不同 domain（如 HalfCheetah、Walker2d、Swimmer 等）
    的多智能体环境包装为统一维度的接口。每个 domain 可包含
    多个 task（如 run_fwd、run_bwd），所有 task 共享同一个
    底层环境实例。

    维度对齐策略（对应论文 Section 3 Multi-task Settings）:
        (1) 观测前拼接 one-hot 任务向量作为任务标签
        (2) 观测末尾零填充至统一维度
        (3) 通过动作掩码屏蔽无效动作维度
        (4) 智能体数量不足时补充零观测虚拟智能体
    """

    def __init__(
        self,
        env_args: dict[str, Any],
        render_mode: str | None = None,
    ) -> None:
        """
        初始化多任务封装器。

        参数:
            env_args: 环境配置字典，必须包含以下键:
                - "envs": 字典，键为 ARGS_REGISTRY 中的
                    配置名（如 "2_Agent_HalfCheetah"），
                    值为该 domain 下的任务名列表
                    （如 ["run_fwd", "run_bwd"]）。

                示例::

                    {
                        "envs": {
                            "2_Agent_HalfCheetah": [
                                "run_fwd", "run_bwd",
                            ],
                            "2_Agent_Walker2d": [
                                "walk_fwd", "run_fwd",
                            ],
                            "2_Agent_Swimmer": [
                                "swim_fwd", "swim_bwd",
                            ],
                        }
                    }
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"，透传给底层环境。
        """
        self._env_args = env_args
        self._render_mode = render_mode
        self._envs = self._make_envs()

        # -- 任务展平与映射 --
        # tasks: 所有任务名的展平列表
        # domain_indices: 每个全局任务对应的 domain 索引
        # env_names: 带 domain 前缀的可读名称列表
        self._tasks: list[str] = []
        self._domain_indices: list[int] = []
        self._env_names: list[str] = []
        for domain_idx, (config_name, tasks) in enumerate(
            env_args["envs"].items()
        ):
            self._tasks.extend(tasks)
            self._domain_indices.extend(
                [domain_idx] * len(tasks)
            )
            self._env_names.extend(
                f"{config_name}_{task}" for task in tasks
            )

        self._n_tasks: int = len(self._tasks)

        # -- 各 domain 环境信息 --
        # agents: 每个 domain 的智能体名称列表
        # obs_shapes: 每个 domain 中各智能体的观测维度
        # act_shapes: 每个 domain 中各智能体的动作维度
        self._domain_agents: list[list[str]] = []
        self._domain_obs_shapes: list[list[int]] = []
        self._domain_act_shapes: list[list[int]] = []
        for env in self._envs:
            agents = env.agents
            self._domain_agents.append(agents)
            self._domain_obs_shapes.append([
                env.observation_spaces[agent].shape[0]
                for agent in agents
            ])
            self._domain_act_shapes.append([
                env.action_spaces[agent].shape[0]
                for agent in agents
            ])

        # -- 统一维度计算 --
        self._n_agents: int = max(
            len(agents) for agents in self._domain_agents
        )
        # 统一智能体名称（用于对外接口的键名）
        self._agents: list[str] = [
            f"agent_{i}" for i in range(self._n_agents)
        ]

        # 原始最大观测维度（不含 one-hot）
        max_obs_dim = max(
            dim
            for shapes in self._domain_obs_shapes
            for dim in shapes
        )
        # 统一观测维度 = one-hot 任务编码 + 最大原始观测维度
        self._obs_size: int = self._n_tasks + max_obs_dim
        # 统一动作维度 = 所有智能体中最大动作维度
        self._act_size: int = max(
            dim
            for shapes in self._domain_act_shapes
            for dim in shapes
        )

        # -- 构建统一空间 --
        self._observation_spaces: dict[str, Box] = {
            agent: Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_size,),
                dtype=np.float64,
            )
            for agent in self._agents
        }
        self._action_spaces: dict[str, Box] = {
            agent: Box(
                low=-1.0,
                high=1.0,
                shape=(self._act_size,),
                dtype=np.float32,
            )
            for agent in self._agents
        }

        # -- 构建列表形式空间（供 VecEnv wrapper 和算法按索引访问）--
        self.observation_space: list[Box] = [
            self._observation_spaces[agent]
            for agent in self._agents
        ]
        self.share_observation_space: list[Box] = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._obs_size,),
                dtype=np.float64,
            )
            for _ in range(self._n_agents)
        ]
        self.action_space: list[Box] = [
            self._action_spaces[agent]
            for agent in self._agents
        ]

        # -- 当前任务状态 --
        self._task_idx: int = 0

    # ----------------------------------------------------------------
    # 公开属性
    # ----------------------------------------------------------------

    @property
    def n_tasks(self) -> int:
        """返回总任务数。"""
        return self._n_tasks

    @property
    def n_agents(self) -> int:
        """返回统一后的最大智能体数量。"""
        return self._n_agents

    @property
    def agents(self) -> list[str]:
        """返回统一后的智能体名称列表。"""
        return self._agents

    @property
    def obs_size(self) -> int:
        """返回统一后的观测维度（含 one-hot 编码）。"""
        return self._obs_size

    @property
    def act_size(self) -> int:
        """返回统一后的动作维度。"""
        return self._act_size

    @property
    def observation_spaces(self) -> dict[str, Box]:
        """返回统一后的观测空间字典。"""
        return self._observation_spaces

    @property
    def action_spaces(self) -> dict[str, Box]:
        """返回统一后的动作空间字典。"""
        return self._action_spaces

    @property
    def task(self) -> str:
        """返回当前任务名称。"""
        return self._tasks[self._task_idx]

    @property
    def task_idx(self) -> int:
        """返回当前全局任务索引。"""
        return self._task_idx

    @property
    def tasks(self) -> list[str]:
        """返回所有任务名称的展平列表。"""
        return self._tasks

    @property
    def domain_idx(self) -> int:
        """返回当前任务对应的 domain 索引。"""
        return self._domain_indices[self._task_idx]

    @property
    def env(self) -> Any:
        """返回当前任务对应的底层环境实例。"""
        return self._envs[self.domain_idx]

    @property
    def env_names(self) -> list[str]:
        """返回所有任务的带 domain 前缀的可读名称列表。"""
        return self._env_names

    # ----------------------------------------------------------------
    # 环境创建
    # ----------------------------------------------------------------

    def _make_envs(self) -> list[Any]:
        """
        根据配置创建所有 domain 的环境实例。

        遍历 env_args["envs"] 中的每个配置名，从
        ARGS_REGISTRY 获取智能体分割参数，从 ENV_REGISTRY
        获取对应的环境类并实例化。

        返回:
            各 domain 环境实例的列表，顺序与配置中的
            键顺序一致。

        异常:
            KeyError: 配置名不在 ARGS_REGISTRY 中，
                或对应的 scenario 不在 ENV_REGISTRY 中。
        """
        envs = []
        for config_name in self._env_args["envs"]:
            if config_name not in ARGS_REGISTRY:
                raise KeyError(
                    f"配置名 {config_name!r} 不在 "
                    f"ARGS_REGISTRY 中，可选: "
                    f"{list(ARGS_REGISTRY.keys())}"
                )
            args = ARGS_REGISTRY[config_name]
            # 从配置名中提取 scenario 名称
            # 如 "2_Agent_HalfCheetah" -> "HalfCheetah"
            scenario = config_name.split("_Agent_", 1)[-1]
            if scenario not in ENV_REGISTRY:
                raise KeyError(
                    f"scenario {scenario!r} 不在 "
                    f"ENV_REGISTRY 中，可选: "
                    f"{list(ENV_REGISTRY.keys())}"
                )
            env_cls = ENV_REGISTRY[scenario]
            env = env_cls(
                agent_conf=args["agent_conf"],
                agent_obsk=args["agent_obsk"],
                render_mode=self._render_mode,
            )
            envs.append(env)
        return envs

    # ----------------------------------------------------------------
    # 观测对齐
    # ----------------------------------------------------------------

    def _pad_obs(
        self,
        raw_obs: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        """
        将底层环境的原始观测对齐为统一维度。

        对齐步骤:
            1. 在观测前拼接 one-hot 任务编码
            2. 观测末尾零填充至统一维度
            3. 智能体数量不足时补充全零虚拟智能体

        参数:
            raw_obs: 底层环境返回的观测字典，键为
                智能体名称，值为各自的观测向量。

        返回:
            统一维度的观测字典，键为 self.agents
            中的名称，值为长度 obs_size 的向量。
        """
        # one-hot 任务编码
        onehot = np.zeros(self._n_tasks, dtype=np.float64)
        onehot[self._task_idx] = 1.0

        padded: dict[str, NDArray] = {}
        real_agents = self._domain_agents[self.domain_idx]

        # 真实智能体：one-hot + 原始观测 + 零填充
        for i, agent_key in enumerate(real_agents):
            obs = raw_obs[agent_key]
            pad_len = self._obs_size - self._n_tasks - len(obs)
            padded[self._agents[i]] = np.concatenate([
                onehot, obs, np.zeros(pad_len, dtype=np.float64),
            ])

        # 虚拟智能体：全零向量
        for i in range(len(real_agents), self._n_agents):
            padded[self._agents[i]] = np.zeros(
                self._obs_size, dtype=np.float64,
            )

        return padded

    # ----------------------------------------------------------------
    # 任务切换
    # ----------------------------------------------------------------

    def set_task(
        self,
        task_idx: int | None = None,
    ) -> int:
        """
        切换当前任务并同步底层环境。

        参数:
            task_idx: 全局任务索引。为 None 时随机选取。

        返回:
            切换后的全局任务索引。

        异常:
            IndexError: 当任务索引超出范围时抛出。
        """
        if task_idx is None:
            task_idx = int(
                np.random.randint(self._n_tasks)
            )
        if not 0 <= task_idx < self._n_tasks:
            raise IndexError(
                f"任务索引 {task_idx} 超出范围"
                f" [0, {self._n_tasks})"
            )
        self._task_idx = task_idx
        # 同步底层环境的任务（切换奖励函数）
        self.env.set_task(self._tasks[task_idx])
        return self._task_idx

    # ----------------------------------------------------------------
    # 动作裁剪
    # ----------------------------------------------------------------

    def _crop_actions(
        self,
        actions: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        """
        将统一维度的动作裁剪为当前环境的实际维度。

        裁剪步骤:
            1. 丢弃虚拟智能体的动作
            2. 截取每个真实智能体动作的前 N 维
            3. 键名映射回底层环境的智能体名称

        参数:
            actions: 统一维度的动作字典，键为
                self.agents 中的名称。

        返回:
            裁剪后的动作字典，键为底层环境的
            智能体名称，值为实际维度的动作向量。
        """
        d_idx = self.domain_idx
        real_agents = self._domain_agents[d_idx]
        act_shapes = self._domain_act_shapes[d_idx]

        cropped: dict[str, NDArray] = {}
        for i, agent_key in enumerate(real_agents):
            cropped[agent_key] = (
                actions[self._agents[i]][:act_shapes[i]]
            )
        return cropped

    # ----------------------------------------------------------------
    # 环境交互
    # ----------------------------------------------------------------

    def seed(
        self,
        seed: int,
    ) -> None:
        """
        设置随机种子，将在下次 reset 时生效。

        为兼容旧版 gym 的 env.seed() 调用方式，种子
        会被暂存，并在 reset 时透传给底层环境。

        参数:
            seed: 随机种子。
        """
        self._seed = int(seed)

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[list[NDArray], list[NDArray], list]:
        """
        重置当前任务的底层环境并返回统一维度的观测。

        返回格式与 VecEnv wrapper 对齐:
        (obs_n, share_obs_n, available_actions)

        参数:
            seed: 随机种子，透传给底层环境。若为 None
                则使用 seed() 方法暂存的种子。

        返回:
            (obs_n, share_obs_n, available_actions)
            三元组，其中:
            - obs_n: 各智能体观测列表。
            - share_obs_n: 各智能体共享观测列表。
            - available_actions: 可用动作列表（连续
              空间下为 [None, ...]）。
        """
        if seed is None:
            seed = getattr(self, "_seed", None)
            self._seed = None
        raw_obs, _ = self.env.reset(seed=seed)
        obs = self._pad_obs(raw_obs)

        # 转为列表形式
        obs_n = [
            obs[agent] for agent in self._agents
        ]
        share_obs_n = [
            np.zeros(1, dtype=np.float32)
            for _ in range(self._n_agents)
        ]
        available_actions = [
            None for _ in range(self._n_agents)
        ]
        return obs_n, share_obs_n, available_actions

    def step(
        self,
        actions: list[NDArray],
    ) -> tuple[
        list[NDArray], list[NDArray],
        list[NDArray], list[bool],
        list[dict], list,
    ]:
        """
        执行一步交互并返回统一维度的结果。

        返回格式与 VecEnv wrapper 对齐:
        (obs_n, share_obs_n, reward_n, done_n,
         info_n, available_actions)

        参数:
            actions: 各智能体动作列表，每个元素为
                统一维度的动作向量。

        返回:
            (obs_n, share_obs_n, reward_n, done_n,
             info_n, available_actions) 六元组。
        """
        # 将列表转为字典并裁剪
        actions_dict: dict[str, NDArray] = {}
        for i, agent in enumerate(self._agents):
            actions_dict[agent] = actions[i]
        cropped = self._crop_actions(actions_dict)

        raw_obs, raw_rew, raw_term, raw_trunc, raw_info = (
            self.env.step(cropped)
        )

        # 对齐观测
        obs = self._pad_obs(raw_obs)

        # 映射真实智能体 + 补充虚拟智能体
        real_agents = (
            self._domain_agents[self.domain_idx]
        )

        obs_n: list[NDArray] = []
        reward_n: list[NDArray] = []
        done_n: list[bool] = []
        info_n: list[dict] = []

        for i, agent in enumerate(self._agents):
            obs_n.append(obs[agent])
            if i < len(real_agents):
                agent_key = real_agents[i]
                reward_n.append(
                    np.array([raw_rew[agent_key]]),
                )
                done = (
                    raw_term[agent_key]
                    or raw_trunc[agent_key]
                )
                done_n.append(done)
                info_i = dict(raw_info[agent_key])
                info_i["bad_transition"] = (
                    raw_trunc[agent_key]
                    and not raw_term[agent_key]
                )
                info_n.append(info_i)
            else:
                reward_n.append(
                    np.array([0.0]),
                )
                done_n.append(False)
                info_n.append(
                    {"bad_transition": False},
                )

        share_obs_n = [
            np.zeros(1, dtype=np.float32)
            for _ in range(self._n_agents)
        ]
        available_actions = [
            None for _ in range(self._n_agents)
        ]

        return (
            obs_n, share_obs_n, reward_n,
            done_n, info_n, available_actions,
        )

    # ----------------------------------------------------------------
    # 动作掩码
    # ----------------------------------------------------------------

    def get_action_mask(self) -> NDArray:
        """
        生成所有任务的动作掩码矩阵。

        掩码形状为 (n_tasks, n_agents, act_size)，
        有效动作维度为 1.0，填充维度和虚拟智能体为 0.0。

        返回:
            动作掩码数组。
        """
        mask = np.zeros(
            (self._n_tasks, self._n_agents, self._act_size),
            dtype=np.float32,
        )
        for task_i in range(self._n_tasks):
            d_idx = self._domain_indices[task_i]
            act_shapes = self._domain_act_shapes[d_idx]
            for agent_i, act_dim in enumerate(act_shapes):
                mask[task_i, agent_i, :act_dim] = 1.0
        return mask

    # ----------------------------------------------------------------
    # 资源管理
    # ----------------------------------------------------------------

    def close(self) -> None:
        """关闭所有底层环境，释放资源。"""
        for env in self._envs:
            env.close()

    def render(self) -> Any:
        """透传渲染到当前任务对应的底层环境。"""
        return self.env.render()
