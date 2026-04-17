"""离线策略经验回放缓冲区基类模块。"""

import numpy as np

from baselines.utils.env import (
    get_shape_from_act_space,
    get_shape_from_obs_space,
)


class OffPolicyBufferBase:
    """离线策略经验回放缓冲区的基类。"""

    def __init__(
        self,
        args,
        share_obs_space,
        num_agents,
        obs_spaces,
        act_spaces,
    ):
        """初始化离线策略缓冲区。

        参数:
            args: 参数字典
            share_obs_space: 共享观测空间
            num_agents: 智能体数量
            obs_spaces: 各智能体的观测空间
            act_spaces: 各智能体的动作空间
        """
        self.buffer_size = args["buffer_size"]
        self.batch_size = args["batch_size"]
        self.n_step = args["n_step"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.gamma = args["gamma"]
        self.cur_size = 0  # 当前已占用的缓冲区大小
        self.idx = 0  # 当前插入位置的索引
        self.num_agents = num_agents
        self.act_spaces = act_spaces

        # 获取共享观测、观测和动作的形状
        self.share_obs_shape = get_shape_from_obs_space(
            share_obs_space
        )
        if isinstance(self.share_obs_shape[-1], list):
            self.share_obs_shape = self.share_obs_shape[:1]
        obs_shapes = []
        act_shapes = []
        for agent_id in range(num_agents):
            obs_shape = get_shape_from_obs_space(
                obs_spaces[agent_id]
            )
            if isinstance(obs_shape[-1], list):
                obs_shape = obs_shape[:1]
            obs_shapes.append(obs_shape)
            act_shapes.append(
                get_shape_from_act_space(act_spaces[agent_id])
            )

        # 各智能体的观测和下一步观测缓冲区
        self.obs = []
        self.next_obs = []
        for agent_id in range(num_agents):
            self.obs.append(
                np.zeros(
                    (self.buffer_size, *obs_shapes[agent_id]),
                    dtype=np.float32,
                )
            )
            self.next_obs.append(
                np.zeros(
                    (self.buffer_size, *obs_shapes[agent_id]),
                    dtype=np.float32,
                )
            )

        # 各智能体的有效转移标记缓冲区
        self.valid_transitions = []
        for agent_id in range(num_agents):
            self.valid_transitions.append(
                np.ones(
                    (self.buffer_size, 1),
                    dtype=np.float32,
                )
            )

        # 各智能体每步的动作及可用动作缓冲区
        self.actions = []
        self.available_actions = []
        self.next_available_actions = []
        for agent_id in range(num_agents):
            self.actions.append(
                np.zeros(
                    (self.buffer_size, act_shapes[agent_id]),
                    dtype=np.float32,
                )
            )
            act_cls = act_spaces[agent_id].__class__.__name__
            if act_cls == "Discrete":
                self.available_actions.append(
                    np.zeros(
                        (
                            self.buffer_size,
                            act_spaces[agent_id].n,
                        ),
                        dtype=np.float32,
                    )
                )
                self.next_available_actions.append(
                    np.zeros(
                        (
                            self.buffer_size,
                            act_spaces[agent_id].n,
                        ),
                        dtype=np.float32,
                    )
                )

    def insert(
        self,
        data,
    ):
        """向缓冲区中插入数据。

        参数:
            data: 元组，包含以下元素：
                share_obs, obs, actions,
                available_actions, reward, done,
                valid_transitions, term,
                next_share_obs, next_obs,
                next_available_actions
        """
        (
            share_obs,
            obs,
            actions,
            available_actions,
            reward,
            done,
            valid_transitions,
            term,
            next_share_obs,
            next_obs,
            next_available_actions,
        ) = data
        length = share_obs.shape[0]
        if self.idx + length <= self.buffer_size:
            # 未溢出
            s = self.idx
            e = self.idx + length
            self.share_obs[s:e] = share_obs.copy()
            self.rewards[s:e] = reward.copy()
            self.dones[s:e] = done.copy()
            self.terms[s:e] = term.copy()
            self.next_share_obs[s:e] = (
                next_share_obs.copy()
            )
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = (
                    obs[agent_id].copy()
                )
                self.actions[agent_id][s:e] = (
                    actions[agent_id].copy()
                )
                self.valid_transitions[agent_id][s:e] = (
                    valid_transitions[agent_id].copy()
                )
                act_cls = (
                    self.act_spaces[agent_id]
                    .__class__.__name__
                )
                if act_cls == "Discrete":
                    self.available_actions[agent_id][
                        s:e
                    ] = available_actions[agent_id].copy()
                    self.next_available_actions[
                        agent_id
                    ][s:e] = (
                        next_available_actions[
                            agent_id
                        ].copy()
                    )
                self.next_obs[agent_id][s:e] = (
                    next_obs[agent_id].copy()
                )
        else:
            # 溢出处理
            len1 = self.buffer_size - self.idx  # 第一段长度
            len2 = length - len1  # 第二段长度

            # 插入第一段
            s = self.idx
            e = self.buffer_size
            self.share_obs[s:e] = (
                share_obs[0:len1].copy()
            )
            self.rewards[s:e] = reward[0:len1].copy()
            self.dones[s:e] = done[0:len1].copy()
            self.terms[s:e] = term[0:len1].copy()
            self.next_share_obs[s:e] = (
                next_share_obs[0:len1].copy()
            )
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = (
                    obs[agent_id][0:len1].copy()
                )
                self.actions[agent_id][s:e] = (
                    actions[agent_id][0:len1].copy()
                )
                self.valid_transitions[agent_id][
                    s:e
                ] = (
                    valid_transitions[agent_id][
                        0:len1
                    ].copy()
                )
                act_cls = (
                    self.act_spaces[agent_id]
                    .__class__.__name__
                )
                if act_cls == "Discrete":
                    self.available_actions[agent_id][
                        s:e
                    ] = (
                        available_actions[agent_id][
                            0:len1
                        ].copy()
                    )
                    self.next_available_actions[
                        agent_id
                    ][s:e] = (
                        next_available_actions[
                            agent_id
                        ][0:len1].copy()
                    )
                self.next_obs[agent_id][s:e] = (
                    next_obs[agent_id][0:len1].copy()
                )

            # 插入第二段
            s = 0
            e = len2
            self.share_obs[s:e] = (
                share_obs[len1:length].copy()
            )
            self.rewards[s:e] = (
                reward[len1:length].copy()
            )
            self.dones[s:e] = (
                done[len1:length].copy()
            )
            self.terms[s:e] = (
                term[len1:length].copy()
            )
            self.next_share_obs[s:e] = (
                next_share_obs[len1:length].copy()
            )
            for agent_id in range(self.num_agents):
                self.obs[agent_id][s:e] = (
                    obs[agent_id][len1:length].copy()
                )
                self.actions[agent_id][s:e] = (
                    actions[agent_id][
                        len1:length
                    ].copy()
                )
                self.valid_transitions[agent_id][
                    s:e
                ] = (
                    valid_transitions[agent_id][
                        len1:length
                    ].copy()
                )
                act_cls = (
                    self.act_spaces[agent_id]
                    .__class__.__name__
                )
                if act_cls == "Discrete":
                    self.available_actions[agent_id][
                        s:e
                    ] = (
                        available_actions[agent_id][
                            len1:length
                        ].copy()
                    )
                    self.next_available_actions[
                        agent_id
                    ][s:e] = (
                        next_available_actions[
                            agent_id
                        ][len1:length].copy()
                    )
                self.next_obs[agent_id][s:e] = (
                    next_obs[agent_id][
                        len1:length
                    ].copy()
                )

        # 更新索引
        self.idx = (
            (self.idx + length) % self.buffer_size
        )
        # 更新当前已用大小
        self.cur_size = min(
            self.cur_size + length,
            self.buffer_size,
        )

    def sample(self):
        """从缓冲区中采样。"""
        pass

    def next(
        self,
        indices,
    ):
        """根据索引获取下一步数据。"""
        pass

    def update_end_flag(self):
        """更新结束标记。"""
        pass

    def get_mean_rewards(self):
        """获取缓冲区中奖励的均值。"""
        return np.mean(self.rewards[: self.cur_size])
