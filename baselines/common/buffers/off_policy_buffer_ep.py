"""离策略经验回放缓冲区模块。"""

import numpy as np
import torch

from baselines.common.buffers.off_policy_buffer_base import (
    OffPolicyBufferBase,
)


class OffPolicyBufferEP(OffPolicyBufferBase):
    """使用环境提供（EP）状态的离策略缓冲区。"""

    def __init__(
        self,
        args,
        share_obs_space,
        num_agents,
        obs_spaces,
        act_spaces,
    ):
        """初始化离策略缓冲区。

        参数:
            args: (dict) 参数字典
            share_obs_space: (gym.Space 或 list) 共享观测空间
            num_agents: (int) 智能体数量
            obs_spaces: (gym.Space 或 list) 观测空间
            act_spaces: (gym.Space) 动作空间
        """
        super(OffPolicyBufferEP, self).__init__(
            args,
            share_obs_space,
            num_agents,
            obs_spaces,
            act_spaces,
        )

        # 共享观测缓冲区
        self.share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape),
            dtype=np.float32,
        )

        # 下一步共享观测缓冲区
        self.next_share_obs = np.zeros(
            (self.buffer_size, *self.share_obs_shape),
            dtype=np.float32,
        )

        # 每个时间步智能体获得的奖励缓冲区
        self.rewards = np.zeros(
            (self.buffer_size, 1),
            dtype=np.float32,
        )

        # 完成标志和终止标志缓冲区
        self.dones = np.full(
            (self.buffer_size, 1),
            False,
        )
        self.terms = np.full(
            (self.buffer_size, 1),
            False,
        )

    def sample(self):
        """采样训练数据。

        返回:
            sp_share_obs: (batch_size, *dim)
            sp_obs: (n_agents, batch_size, *dim)
            sp_actions: (n_agents, batch_size, *dim)
            sp_available_actions:
                (n_agents, batch_size, *dim)
            sp_reward: (batch_size, 1)
            sp_done: (batch_size, 1)
            sp_valid_transitions:
                (n_agents, batch_size, 1)
            sp_term: (batch_size, 1)
            sp_next_share_obs: (batch_size, *dim)
            sp_next_obs:
                (n_agents, batch_size, *dim)
            sp_next_available_actions:
                (n_agents, batch_size, *dim)
            sp_gamma: (batch_size, 1)
        """
        # 更新当前结束标志
        self.update_end_flag()
        # 采样索引，形状: (batch_size, )
        indice = torch.randperm(self.cur_size).numpy()[
            : self.batch_size
        ]

        # 获取起始索引处的数据
        sp_share_obs = self.share_obs[indice]
        sp_obs = np.array(
            [
                self.obs[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        sp_actions = np.array(
            [
                self.actions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        sp_valid_transitions = np.array(
            [
                self.valid_transitions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ]
        )
        if self.act_spaces[0].__class__.__name__ == \
                "Discrete":
            sp_available_actions = np.array(
                [
                    self.available_actions[agent_id][
                        indice
                    ]
                    for agent_id
                    in range(self.num_agents)
                ]
            )

        # 沿 n 步计算索引
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        # 获取最后索引处的数据
        sp_done = self.dones[indices[-1]]
        sp_term = self.terms[indices[-1]]
        sp_next_share_obs = \
            self.next_share_obs[indices[-1]]
        sp_next_obs = np.array(
            [
                self.next_obs[agent_id][indices[-1]]
                for agent_id in range(self.num_agents)
            ]
        )
        if self.act_spaces[0].__class__.__name__ == \
                "Discrete":
            sp_next_available_actions = np.array(
                [
                    self.next_available_actions[
                        agent_id
                    ][indices[-1]]
                    for agent_id
                    in range(self.num_agents)
                ]
            )

        # 计算累积奖励和对应的折扣因子
        gamma_buffer = np.ones(self.n_step + 1)
        for i in range(1, self.n_step + 1):
            gamma_buffer[i] = (
                gamma_buffer[i - 1] * self.gamma
            )
        sp_reward = np.zeros((self.batch_size, 1))
        gammas = np.full(
            self.batch_size,
            self.n_step,
        )
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            gammas[self.end_flag[now] > 0] = n + 1
            sp_reward[
                self.end_flag[now] > 0
            ] = 0.0
            sp_reward = (
                self.rewards[now]
                + self.gamma * sp_reward
            )
        sp_gamma = gamma_buffer[gammas].reshape(
            self.batch_size,
            1,
        )

        if self.act_spaces[0].__class__.__name__ == \
                "Discrete":
            return (
                sp_share_obs,
                sp_obs,
                sp_actions,
                sp_available_actions,
                sp_reward,
                sp_done,
                sp_valid_transitions,
                sp_term,
                sp_next_share_obs,
                sp_next_obs,
                sp_next_available_actions,
                sp_gamma,
            )
        else:
            return (
                sp_share_obs,
                sp_obs,
                sp_actions,
                None,
                sp_reward,
                sp_done,
                sp_valid_transitions,
                sp_term,
                sp_next_share_obs,
                sp_next_obs,
                None,
                sp_gamma,
            )

    def next(
        self,
        indices,
    ):
        """获取下一个索引。"""
        return (
            indices
            + (1 - self.end_flag[indices])
            * self.n_rollout_threads
        ) % self.buffer_size

    def update_end_flag(self):
        """更新用于计算 n 步回报的结束标志。

        结束标志在回合结束的步骤或最新的
        未完成步骤处为 True。
        """
        self.unfinished_index = (
            self.idx
            - np.arange(self.n_rollout_threads)
            - 1
            + self.cur_size
        ) % self.cur_size
        # (batch_size, )
        self.end_flag = (
            self.dones.copy().squeeze()
        )
        self.end_flag[self.unfinished_index] = True
