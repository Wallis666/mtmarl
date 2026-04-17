"""基于环境提供状态（EP）的在线策略评论家缓冲区模块。"""

import numpy as np
import torch

from baselines.utils.env import get_shape_from_obs_space
from baselines.utils.trans import _flatten, _sa_cast


class OnPolicyCriticBufferEP:
    """使用环境提供状态（EP）的在线策略评论家缓冲区。"""

    def __init__(
        self,
        args,
        share_obs_space,
    ):
        """初始化在线策略评论家缓冲区。

        Args:
            args: (dict) 参数字典
            share_obs_space: (gym.Space 或 list) 共享观测空间
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]
        self.use_gae = args["use_gae"]
        self.use_proper_time_limits = (
            args["use_proper_time_limits"]
        )

        share_obs_shape = get_shape_from_obs_space(
            share_obs_space
        )
        if isinstance(share_obs_shape[-1], list):
            share_obs_shape = share_obs_shape[:1]

        # 共享观测缓冲区
        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                *share_obs_shape,
            ),
            dtype=np.float32,
        )

        # 评论家 RNN 隐藏状态缓冲区
        self.rnn_states_critic = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # 评论家价值预测缓冲区
        self.value_preds = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )

        # 每个时间步的回报缓冲区
        self.returns = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )

        # 每个时间步智能体收到的奖励缓冲区
        self.rewards = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )

        # 每个时间步回合是否结束的掩码缓冲区
        self.masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )

        # 区分截断与终止的掩码缓冲区。
        # 若为 0 表示截断；若为 1 且 masks 为 0 表示终止；
        # 否则表示回合尚未结束。
        self.bad_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(
        self,
        share_obs,
        rnn_states_critic,
        value_preds,
        rewards,
        masks,
        bad_masks,
    ):
        """将数据插入缓冲区。"""
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_critic[self.step + 1] = (
            rnn_states_critic.copy()
        )
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.bad_masks[self.step + 1] = bad_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """更新后，将最后一步数据复制到缓冲区首位。"""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_critic[0] = (
            self.rnn_states_critic[-1].copy()
        )
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        """获取平均奖励，用于日志记录。"""
        return np.mean(self.rewards)

    def compute_returns(
        self,
        next_value,
        value_normalizer=None,
    ):
        """计算回报，使用折扣奖励和或 GAE 方法。

        Args:
            next_value: (np.ndarray) 最后一步之后的
                价值预测。
            value_normalizer: (ValueNorm) 若非 None，
                则为 ValueNorm 价值归一化器实例。
        """
        # 区分截断与终止
        if self.use_proper_time_limits:
            if self.use_gae:  # 使用 GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(
                    range(self.rewards.shape[0])
                ):
                    if value_normalizer is not None:
                        # 使用 ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(
                                self.value_preds[step + 1]
                            )
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(
                                self.value_preds[step]
                            )
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.masks[step + 1]
                            * gae
                        )
                        gae = (
                            self.bad_masks[step + 1] * gae
                        )
                        self.returns[step] = (
                            gae
                            + value_normalizer.denormalize(
                                self.value_preds[step]
                            )
                        )
                    else:
                        # 不使用 ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.masks[step + 1]
                            * gae
                        )
                        gae = (
                            self.bad_masks[step + 1] * gae
                        )
                        self.returns[step] = (
                            gae + self.value_preds[step]
                        )
            else:  # 不使用 GAE
                self.returns[-1] = next_value
                for step in reversed(
                    range(self.rewards.shape[0])
                ):
                    if value_normalizer is not None:
                        # 使用 ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1]
                            * self.gamma
                            * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[
                            step + 1
                        ] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:
                        # 不使用 ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1]
                            * self.gamma
                            * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[
                            step + 1
                        ] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:
            # 不区分截断与终止，所有结束回合视为终止
            if self.use_gae:  # 使用 GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(
                    range(self.rewards.shape[0])
                ):
                    if value_normalizer is not None:
                        # 使用 ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(
                                self.value_preds[step + 1]
                            )
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(
                                self.value_preds[step]
                            )
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.masks[step + 1]
                            * gae
                        )
                        self.returns[step] = (
                            gae
                            + value_normalizer.denormalize(
                                self.value_preds[step]
                            )
                        )
                    else:
                        # 不使用 ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma
                            * self.gae_lambda
                            * self.masks[step + 1]
                            * gae
                        )
                        self.returns[step] = (
                            gae + self.value_preds[step]
                        )
            else:  # 不使用 GAE
                self.returns[-1] = next_value
                for step in reversed(
                    range(self.rewards.shape[0])
                ):
                    self.returns[step] = (
                        self.returns[step + 1]
                        * self.gamma
                        * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def feed_forward_generator_critic(
        self,
        critic_num_mini_batch=None,
        mini_batch_size=None,
    ):
        """使用 MLP 网络的评论家训练数据生成器。

        Args:
            critic_num_mini_batch: (int) 评论家的
                小批次数量。
            mini_batch_size: (int) 评论家的小批次大小。
        """
        # 获取 episode_length、n_rollout_threads
        # 和 mini_batch_size
        episode_length, n_rollout_threads = (
            self.rewards.shape[0:2]
        )
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert (
                batch_size >= critic_num_mini_batch
            ), (
                f"进程数 ({n_rollout_threads}) "
                f"* 步数 ({episode_length}) = "
                f"{n_rollout_threads * episode_length} "
                f"必须大于等于评论家小批次数 "
                f"({critic_num_mini_batch})。"
            )
            mini_batch_size = (
                batch_size // critic_num_mini_batch
            )

        # 随机打乱索引
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[
                i * mini_batch_size
                : (i + 1) * mini_batch_size
            ]
            for i in range(critic_num_mini_batch)
        ]

        # 合并前两个维度（episode_length 和
        # n_rollout_threads）形成批次。
        # 以 share_obs 形状为例：
        # (episode_length + 1, n_rollout_threads,
        #  *share_obs_shape)
        # --> (episode_length, n_rollout_threads,
        #      *share_obs_shape)
        # --> (episode_length * n_rollout_threads,
        #      *share_obs_shape)
        share_obs = self.share_obs[:-1].reshape(
            -1, *self.share_obs.shape[2:]
        )
        # 实际未使用，仅为保持一致性
        rnn_states_critic = (
            self.rnn_states_critic[:-1].reshape(
                -1, *self.rnn_states_critic.shape[2:]
            )
        )
        value_preds = self.value_preds[:-1].reshape(
            -1, 1
        )
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            # share_obs 形状：
            # (episode_length * n_rollout_threads,
            #  *share_obs_shape)
            # --> (mini_batch_size, *share_obs_shape)
            share_obs_batch = share_obs[indices]
            rnn_states_critic_batch = (
                rnn_states_critic[indices]
            )
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]

            yield (
                share_obs_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
            )

    def naive_recurrent_generator_critic(
        self,
        critic_num_mini_batch,
    ):
        """使用 RNN 网络的评论家朴素循环训练数据生成器。

        该生成器不将轨迹拆分为块，因此训练效率
        可能低于 recurrent_generator_critic。

        Args:
            critic_num_mini_batch: (int) 评论家的
                小批次数量。
        """
        # 获取 n_rollout_threads 和 num_envs_per_batch
        n_rollout_threads = self.rewards.shape[1]
        assert (
            n_rollout_threads >= critic_num_mini_batch
        ), (
            f"进程数 ({n_rollout_threads}) "
            f"必须大于等于小批次数 "
            f"({critic_num_mini_batch})。"
        )
        num_envs_per_batch = (
            n_rollout_threads // critic_num_mini_batch
        )

        # 随机打乱索引
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(critic_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[
                start_id: start_id + num_envs_per_batch
            ]
            share_obs_batch = _flatten(
                T, N, self.share_obs[:-1, ids]
            )
            value_preds_batch = _flatten(
                T, N, self.value_preds[:-1, ids]
            )
            return_batch = _flatten(
                T, N, self.returns[:-1, ids]
            )
            masks_batch = _flatten(
                T, N, self.masks[:-1, ids]
            )
            rnn_states_critic_batch = (
                self.rnn_states_critic[0, ids]
            )

            yield (
                share_obs_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
            )

    def recurrent_generator_critic(
        self,
        critic_num_mini_batch,
        data_chunk_length,
    ):
        """使用 RNN 网络的评论家循环训练数据生成器。

        该生成器将轨迹拆分为长度为 data_chunk_length
        的块，因此训练效率可能高于
        naive_recurrent_generator_critic。

        Args:
            critic_num_mini_batch: (int) 评论家的
                小批次数量。
            data_chunk_length: (int) 数据块长度。
        """
        # 获取 episode_length、n_rollout_threads
        # 和 mini_batch_size
        episode_length, n_rollout_threads = (
            self.rewards.shape[0:2]
        )
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = (
            data_chunks // critic_num_mini_batch
        )

        assert episode_length % data_chunk_length == 0, (
            f"回合长度 ({episode_length}) 必须是"
            f"数据块长度 ({data_chunk_length}) 的整数倍。"
        )
        assert data_chunks >= 2, "需要更大的批次大小"

        # 随机打乱索引
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[
                i * mini_batch_size
                : (i + 1) * mini_batch_size
            ]
            for i in range(critic_num_mini_batch)
        ]

        # 以下数据操作先将数据前两个维度
        # (episode_length, n_rollout_threads) 转置为
        # (n_rollout_threads, episode_length)，再将数据
        # 重塑为 (n_rollout_threads * episode_length,
        # *dim)。
        # 以 share_obs 形状为例：
        # (episode_length + 1, n_rollout_threads,
        #  *share_obs_shape)
        # --> (episode_length, n_rollout_threads,
        #      *share_obs_shape)
        # --> (n_rollout_threads, episode_length,
        #      *share_obs_shape)
        # --> (n_rollout_threads * episode_length,
        #      *share_obs_shape)
        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(
                    -1, *self.share_obs.shape[2:]
                )
            )
        else:
            share_obs = _sa_cast(self.share_obs[:-1])
        value_preds = _sa_cast(self.value_preds[:-1])
        returns = _sa_cast(self.returns[:-1])
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(
                -1,
                *self.rnn_states_critic.shape[2:],
            )
        )

        # 生成小批次
        for indices in sampler:
            share_obs_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(
                    share_obs[ind: ind + data_chunk_length]
                )
                value_preds_batch.append(
                    value_preds[
                        ind: ind + data_chunk_length
                    ]
                )
                return_batch.append(
                    returns[
                        ind: ind + data_chunk_length
                    ]
                )
                masks_batch.append(
                    masks[ind: ind + data_chunk_length]
                )
                # 仅需要起始 RNN 状态
                rnn_states_critic_batch.append(
                    rnn_states_critic[ind]
                )

            L, N = data_chunk_length, mini_batch_size
            # 以下均为形状
            # (data_chunk_length, mini_batch_size, *dim)
            # 的 ndarray
            share_obs_batch = np.stack(
                share_obs_batch, axis=1
            )
            value_preds_batch = np.stack(
                value_preds_batch, axis=1
            )
            return_batch = np.stack(
                return_batch, axis=1
            )
            masks_batch = np.stack(
                masks_batch, axis=1
            )
            # rnn_states_critic_batch 为形状
            # (mini_batch_size, *dim) 的 ndarray
            rnn_states_critic_batch = np.stack(
                rnn_states_critic_batch
            ).reshape(
                N,
                *self.rnn_states_critic.shape[2:],
            )

            # 将 (data_chunk_length, mini_batch_size,
            # *dim) 展平为
            # (data_chunk_length * mini_batch_size, *dim)
            share_obs_batch = _flatten(
                L, N, share_obs_batch
            )
            value_preds_batch = _flatten(
                L, N, value_preds_batch
            )
            return_batch = _flatten(
                L, N, return_batch
            )
            masks_batch = _flatten(
                L, N, masks_batch
            )

            yield (
                share_obs_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
            )
