"""离策略经验回放缓冲区模块。"""

import numpy as np
import torch

from baselines.common.buffers.off_policy_buffer_base import (
    OffPolicyBufferBase,
)


class OffPolicyBufferFP(OffPolicyBufferBase):
    """使用特征裁剪（FP）状态的离策略缓冲区。

    当使用 FP 状态时，评论家对不同的智能体采用不同的全局状态
    作为输入。因此，OffPolicyBufferFP 额外增加了一个智能体
    数量的维度。
    """

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
            args: 参数字典
            share_obs_space: 共享观测空间
            num_agents: 智能体数量
            obs_spaces: 观测空间
            act_spaces: 动作空间
        """
        super(OffPolicyBufferFP, self).__init__(
            args,
            share_obs_space,
            num_agents,
            obs_spaces,
            act_spaces,
        )

        # 共享观测缓冲区
        self.share_obs = np.zeros(
            (
                self.buffer_size,
                self.num_agents,
                *self.share_obs_shape,
            ),
            dtype=np.float32,
        )

        # 下一步共享观测缓冲区
        self.next_share_obs = np.zeros(
            (
                self.buffer_size,
                self.num_agents,
                *self.share_obs_shape,
            ),
            dtype=np.float32,
        )

        # 每个时间步各智能体获得的奖励缓冲区
        self.rewards = np.zeros(
            (self.buffer_size, self.num_agents, 1),
            dtype=np.float32,
        )

        # 结束标志和终止标志缓冲区
        self.dones = np.full(
            (self.buffer_size, self.num_agents, 1),
            False,
        )
        self.terms = np.full(
            (self.buffer_size, self.num_agents, 1),
            False,
        )

    def sample(self):
        """采样训练数据。

        返回:
            sp_share_obs:
                (n_agents * batch_size, *dim)
            sp_obs:
                (n_agents, batch_size, *dim)
            sp_actions:
                (n_agents, batch_size, *dim)
            sp_available_actions:
                (n_agents, batch_size, *dim)
            sp_reward:
                (n_agents * batch_size, 1)
            sp_done:
                (n_agents * batch_size, 1)
            sp_valid_transitions:
                (n_agents, batch_size, 1)
            sp_term:
                (n_agents * batch_size, 1)
            sp_next_share_obs:
                (n_agents * batch_size, *dim)
            sp_next_obs:
                (n_agents, batch_size, *dim)
            sp_next_available_actions:
                (n_agents, batch_size, *dim)
            sp_gamma:
                (n_agents * batch_size, 1)
        """
        # 更新当前结束标志
        self.update_end_flag()
        # 采样索引，形状: (batch_size, )
        indice = torch.randperm(
            self.cur_size,
        ).numpy()[: self.batch_size]

        # 获取起始索引处的数据
        # (batch_size, n_agents, *dim)
        # -> (n_agents, batch_size, *dim)
        # -> (n_agents * batch_size, *dim)
        sp_share_obs = np.concatenate(
            self.share_obs[indice].transpose(1, 0, 2),
            axis=0,
        )
        sp_obs = np.array(
            [
                self.obs[agent_id][indice]
                for agent_id in range(self.num_agents)
            ],
        )
        sp_actions = np.array(
            [
                self.actions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ],
        )
        sp_valid_transitions = np.array(
            [
                self.valid_transitions[agent_id][indice]
                for agent_id in range(self.num_agents)
            ],
        )
        if self.act_spaces[0].__class__.__name__ == \
                'Discrete':
            sp_available_actions = np.array(
                [
                    self.available_actions[agent_id][indice]
                    for agent_id
                    in range(self.num_agents)
                ],
            )

        # 沿 n 步计算索引
        # (batch_size, n_agents)
        indice = np.repeat(
            np.expand_dims(indice, axis=-1),
            self.num_agents,
            axis=-1,
        )
        indices = [indice]
        for _ in range(self.n_step - 1):
            indices.append(self.next(indices[-1]))

        # 获取最后索引处的数据
        # (n_agents, batch_size, 1)
        # -> (n_agents * batch_size, 1)
        sp_done = np.concatenate(
            [
                self.dones[
                    indices[-1][:, agent_id], agent_id
                ]
                for agent_id in range(self.num_agents)
            ],
        )
        # (n_agents, batch_size, 1)
        # -> (n_agents * batch_size, 1)
        sp_term = np.concatenate(
            [
                self.terms[
                    indices[-1][:, agent_id], agent_id
                ]
                for agent_id in range(self.num_agents)
            ],
        )
        # (n_agents, batch_size, *dim)
        # -> (n_agents * batch_size, *dim)
        sp_next_share_obs = np.concatenate(
            [
                self.next_share_obs[
                    indices[-1][:, agent_id], agent_id
                ]
                for agent_id in range(self.num_agents)
            ],
        )
        sp_next_obs = np.array(
            [
                self.next_obs[agent_id][
                    indices[-1][:, agent_id]
                ]
                for agent_id in range(self.num_agents)
            ],
        )
        if self.act_spaces[0].__class__.__name__ == \
                'Discrete':
            sp_next_available_actions = np.array(
                [
                    self.next_available_actions[
                        agent_id
                    ][indices[-1][:, agent_id]]
                    for agent_id
                    in range(self.num_agents)
                ],
            )

        # 计算累积奖励和对应的折扣因子
        gamma_buffer = np.ones(
            (self.num_agents, self.n_step + 1),
        )
        for i in range(1, self.n_step + 1):
            gamma_buffer[:, i] = (
                gamma_buffer[:, i - 1] * self.gamma
            )
        sp_reward = np.zeros(
            (self.batch_size, self.num_agents, 1),
        )
        gammas = np.full(
            (self.batch_size, self.num_agents),
            self.n_step,
        )
        for n in range(self.n_step - 1, -1, -1):
            now = indices[n]
            end_flag = np.column_stack(
                [
                    self.end_flag[
                        now[:, agent_id], agent_id
                    ]
                    for agent_id
                    in range(self.num_agents)
                ],
            )
            gammas[end_flag > 0] = n + 1
            sp_reward[end_flag > 0] = 0.0
            rewards = np.expand_dims(
                np.column_stack(
                    [
                        self.rewards[
                            now[:, agent_id],
                            agent_id,
                        ]
                        for agent_id
                        in range(self.num_agents)
                    ],
                ),
                axis=-1,
            )
            sp_reward = (
                rewards + self.gamma * sp_reward
            )
        sp_reward = np.concatenate(
            sp_reward.transpose(1, 0, 2),
            axis=0,
        )
        # (n_agents * batch_size, )
        # -> (n_agents * batch_size, 1)
        sp_gamma = np.concatenate(
            [
                gamma_buffer[agent_id][
                    gammas[:, agent_id]
                ]
                for agent_id in range(self.num_agents)
            ],
        ).reshape(-1, 1)

        if self.act_spaces[0].__class__.__name__ == \
                'Discrete':
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
        # (batch_size, n_agents)
        end_flag = np.column_stack(
            [
                self.end_flag[
                    indices[:, agent_id], agent_id
                ]
                for agent_id in range(self.num_agents)
            ],
        )
        return (
            indices
            + (1 - end_flag) * self.n_rollout_threads
        ) % self.buffer_size

    def update_end_flag(self):
        """更新当前结束标志以计算 n 步回报。

        结束标志在回合结束的步骤或最新但未完成的步骤处
        为 True。
        """
        self.unfinished_index = (
            self.idx
            - np.arange(self.n_rollout_threads)
            - 1
            + self.cur_size
        ) % self.cur_size
        # FP: (batch_size, n_agents)
        self.end_flag = (
            self.dones.copy().squeeze()
        )
        self.end_flag[
            self.unfinished_index, :
        ] = True
