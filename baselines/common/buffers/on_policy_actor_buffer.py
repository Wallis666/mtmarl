"""同策略（on-policy）演员缓冲区模块。"""

import numpy as np
import torch

from baselines.utils.env import (
    get_shape_from_act_space,
    get_shape_from_obs_space,
)
from baselines.utils.trans import _flatten, _sa_cast


class OnPolicyActorBuffer:
    """同策略演员数据存储缓冲区。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
    ):
        """初始化同策略演员缓冲区。

        参数:
            args: (dict) 参数字典
            obs_space: (gym.Space 或 list) 观测空间
            act_space: (gym.Space) 动作空间
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]

        obs_shape = get_shape_from_obs_space(obs_space)

        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        # 该演员的观测缓冲区
        self.obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                *obs_shape,
            ),
            dtype=np.float32,
        )

        # 该演员的 RNN 状态缓冲区
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # 该演员的可用动作缓冲区
        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    act_space.n,
                ),
                dtype=np.float32,
            )
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        # 该演员的动作缓冲区
        self.actions = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                act_shape,
            ),
            dtype=np.float32,
        )

        # 该演员的动作对数概率缓冲区
        self.action_log_probs = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                act_shape,
            ),
            dtype=np.float32,
        )

        # 该演员的掩码缓冲区。
        # 掩码指示在哪个时间步需要重置 RNN 状态。
        self.masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )

        # 该演员的活跃掩码缓冲区。
        # 活跃掩码指示智能体是否存活。
        self.active_masks = np.ones_like(self.masks)

        self.factor = None

        self.step = 0

    def update_factor(
        self,
        factor,
    ):
        """保存该演员的因子。"""
        self.factor = factor.copy()

    def insert(
        self,
        obs,
        rnn_states,
        actions,
        action_log_probs,
        masks,
        active_masks=None,
        available_actions=None,
    ):
        """将数据插入演员缓冲区。"""
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = (
            action_log_probs.copy()
        )
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = (
                active_masks.copy()
            )
        if available_actions is not None:
            self.available_actions[self.step + 1] = (
                available_actions.copy()
            )

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """更新后，将最后一步的数据复制到缓冲区首位。"""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = (
            self.active_masks[-1].copy()
        )
        if self.available_actions is not None:
            self.available_actions[0] = (
                self.available_actions[-1].copy()
            )

    def feed_forward_generator_actor(
        self,
        advantages,
        actor_num_mini_batch=None,
        mini_batch_size=None,
    ):
        """使用 MLP 网络的演员训练数据生成器。"""

        # 获取 episode_length、n_rollout_threads
        # 和 mini_batch_size
        episode_length, n_rollout_threads = (
            self.actions.shape[0:2]
        )
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                f"进程数 ({n_rollout_threads}) "
                f"* 步数 ({episode_length}) = "
                f"{n_rollout_threads * episode_length}"
                f" 必须大于等于演员小批量数 "
                f"({actor_num_mini_batch})。"
            )
            mini_batch_size = (
                batch_size // actor_num_mini_batch
            )

        # 打乱索引
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[
                i * mini_batch_size
                : (i + 1) * mini_batch_size
            ]
            for i in range(actor_num_mini_batch)
        ]

        # 合并前两个维度（episode_length 和
        # n_rollout_threads）形成批量。
        # 以 obs 形状为例:
        # (episode_length + 1, n_rollout_threads,
        #  *obs_shape)
        # --> (episode_length, n_rollout_threads,
        #      *obs_shape)
        # --> (episode_length * n_rollout_threads,
        #      *obs_shape)
        obs = self.obs[:-1].reshape(
            -1, *self.obs.shape[2:]
        )
        # 实际未使用，仅为保持一致性
        rnn_states = self.rnn_states[:-1].reshape(
            -1, *self.rnn_states.shape[2:]
        )
        actions = self.actions.reshape(
            -1, self.actions.shape[-1]
        )
        if self.available_actions is not None:
            available_actions = (
                self.available_actions[:-1].reshape(
                    -1,
                    self.available_actions.shape[-1],
                )
            )
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = (
            self.active_masks[:-1].reshape(-1, 1)
        )
        action_log_probs = (
            self.action_log_probs.reshape(
                -1, self.action_log_probs.shape[-1]
            )
        )
        if self.factor is not None:
            factor = self.factor.reshape(
                -1, self.factor.shape[-1]
            )
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs 形状:
            # (episode_length * n_rollout_threads,
            #  *obs_shape)
            # --> (mini_batch_size, *obs_shape)
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = (
                    available_actions[indices]
                )
            else:
                available_actions_batch = None
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = (
                action_log_probs[indices]
            )
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.factor is None:
                yield (
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                )
            else:
                factor_batch = factor[indices]
                yield (
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                    factor_batch,
                )

    def naive_recurrent_generator_actor(
        self,
        advantages,
        actor_num_mini_batch,
    ):
        """使用 RNN 网络的演员训练数据生成器。

        该生成器不会将轨迹拆分为块，因此在训练中
        可能不如 recurrent_generator_actor 高效。
        """

        # 获取 n_rollout_threads 和 num_envs_per_batch
        n_rollout_threads = self.actions.shape[1]
        assert (
            n_rollout_threads >= actor_num_mini_batch
        ), (
            f"进程数 ({n_rollout_threads}) "
            f"必须大于等于小批量数 "
            f"({actor_num_mini_batch})。"
        )
        num_envs_per_batch = (
            n_rollout_threads // actor_num_mini_batch
        )

        # 打乱索引
        perm = torch.randperm(
            n_rollout_threads
        ).numpy()

        T, N = self.episode_length, num_envs_per_batch

        # 为每个小批量准备数据
        for batch_id in range(actor_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[
                start_id
                : start_id + num_envs_per_batch
            ]
            obs_batch = _flatten(
                T, N, self.obs[:-1, ids]
            )
            actions_batch = _flatten(
                T, N, self.actions[:, ids]
            )
            masks_batch = _flatten(
                T, N, self.masks[:-1, ids]
            )
            active_masks_batch = _flatten(
                T, N, self.active_masks[:-1, ids]
            )
            old_action_log_probs_batch = _flatten(
                T, N, self.action_log_probs[:, ids]
            )
            adv_targ = _flatten(
                T, N, advantages[:, ids]
            )
            if self.available_actions is not None:
                available_actions_batch = _flatten(
                    T,
                    N,
                    self.available_actions[:-1, ids],
                )
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(
                    T, N, self.factor[:, ids]
                )
            rnn_states_batch = (
                self.rnn_states[0, ids]
            )

            if self.factor is not None:
                yield (
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                    factor_batch,
                )
            else:
                yield (
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                )

    def recurrent_generator_actor(
        self,
        advantages,
        actor_num_mini_batch,
        data_chunk_length,
    ):
        """使用 RNN 网络的演员训练数据生成器。

        该生成器将轨迹拆分为长度为 data_chunk_length
        的块，因此在训练中可能比
        naive_recurrent_generator_actor 更高效。
        """

        # 获取 episode_length、n_rollout_threads
        # 和 mini_batch_size
        episode_length, n_rollout_threads = (
            self.actions.shape[0:2]
        )
        batch_size = (
            n_rollout_threads * episode_length
        )
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = (
            data_chunks // actor_num_mini_batch
        )

        assert episode_length % data_chunk_length == 0, (
            f"回合长度 ({episode_length}) 必须是"
            f"数据块长度 ({data_chunk_length}) "
            f"的整数倍。"
        )
        assert data_chunks >= 2, "需要更大的批量大小"

        # 打乱索引
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[
                i * mini_batch_size
                : (i + 1) * mini_batch_size
            ]
            for i in range(actor_num_mini_batch)
        ]

        # 以下数据操作先将数据前两个维度
        # (episode_length, n_rollout_threads) 转置为
        # (n_rollout_threads, episode_length)，
        # 再将数据重塑为
        # (n_rollout_threads * episode_length, *dim)。
        # 以 obs 形状为例:
        # (episode_length + 1, n_rollout_threads,
        #  *obs_shape)
        # --> (episode_length, n_rollout_threads,
        #      *obs_shape)
        # --> (n_rollout_threads, episode_length,
        #      *obs_shape)
        # --> (n_rollout_threads * episode_length,
        #      *obs_shape)
        if len(self.obs.shape) > 3:
            obs = (
                self.obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.obs.shape[2:])
            )
        else:
            obs = _sa_cast(self.obs[:-1])
        actions = _sa_cast(self.actions)
        action_log_probs = _sa_cast(
            self.action_log_probs
        )
        advantages = _sa_cast(advantages)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(
            self.active_masks[:-1]
        )
        if self.factor is not None:
            factor = _sa_cast(self.factor)
        rnn_states = (
            self.rnn_states[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states.shape[2:])
        )
        if self.available_actions is not None:
            available_actions = _sa_cast(
                self.available_actions[:-1]
            )

        # 生成小批量
        for indices in sampler:
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []

            for index in indices:
                ind = index * data_chunk_length
                obs_batch.append(
                    obs[ind : ind + data_chunk_length]
                )
                actions_batch.append(
                    actions[
                        ind
                        : ind + data_chunk_length
                    ]
                )
                if self.available_actions is not None:
                    available_actions_batch.append(
                        available_actions[
                            ind
                            : ind + data_chunk_length
                        ]
                    )
                masks_batch.append(
                    masks[
                        ind
                        : ind + data_chunk_length
                    ]
                )
                active_masks_batch.append(
                    active_masks[
                        ind
                        : ind + data_chunk_length
                    ]
                )
                old_action_log_probs_batch.append(
                    action_log_probs[
                        ind
                        : ind + data_chunk_length
                    ]
                )
                adv_targ.append(
                    advantages[
                        ind
                        : ind + data_chunk_length
                    ]
                )
                # 只需要起始 RNN 状态
                rnn_states_batch.append(
                    rnn_states[ind]
                )
                if self.factor is not None:
                    factor_batch.append(
                        factor[
                            ind
                            : ind + data_chunk_length
                        ]
                    )

            L, N = data_chunk_length, mini_batch_size
            # 以下均为形状
            # (data_chunk_length, mini_batch_size,
            #  *dim) 的 ndarray
            obs_batch = np.stack(
                obs_batch, axis=1
            )
            actions_batch = np.stack(
                actions_batch, axis=1
            )
            if self.available_actions is not None:
                available_actions_batch = np.stack(
                    available_actions_batch, axis=1
                )
            if self.factor is not None:
                factor_batch = np.stack(
                    factor_batch, axis=1
                )
            masks_batch = np.stack(
                masks_batch, axis=1
            )
            active_masks_batch = np.stack(
                active_masks_batch, axis=1
            )
            old_action_log_probs_batch = np.stack(
                old_action_log_probs_batch, axis=1
            )
            adv_targ = np.stack(adv_targ, axis=1)
            # rnn_states_batch 是形状为
            # (mini_batch_size, *dim) 的 ndarray
            rnn_states_batch = np.stack(
                rnn_states_batch
            ).reshape(
                N, *self.rnn_states.shape[2:]
            )

            # 将 (data_chunk_length,
            # mini_batch_size, *dim) 的 ndarray
            # 展平为 (data_chunk_length
            # * mini_batch_size, *dim)
            obs_batch = _flatten(
                L, N, obs_batch
            )
            actions_batch = _flatten(
                L, N, actions_batch
            )
            if self.available_actions is not None:
                available_actions_batch = _flatten(
                    L, N, available_actions_batch
                )
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(
                    L, N, factor_batch
                )
            masks_batch = _flatten(
                L, N, masks_batch
            )
            active_masks_batch = _flatten(
                L, N, active_masks_batch
            )
            old_action_log_probs_batch = _flatten(
                L, N, old_action_log_probs_batch
            )
            adv_targ = _flatten(L, N, adv_targ)
            if self.factor is not None:
                yield (
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                    factor_batch,
                )
            else:
                yield (
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    available_actions_batch,
                )
