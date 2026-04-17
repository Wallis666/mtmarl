"""HAPPO 算法模块。"""

import numpy as np
import torch
import torch.nn as nn

from baselines.algos.actors.on_policy_base import OnPolicyBase
from baselines.utils.env import check
from baselines.utils.model import get_grad_norm


class HAPPO(OnPolicyBase):
    """HAPPO 算法。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 HAPPO 算法。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间。
            device: 用于张量运算的设备。
        """
        super(HAPPO, self).__init__(
            args, obs_space, act_space, device,
        )

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = (
            args["actor_num_mini_batch"]
        )
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

    def update(
        self,
        sample,
    ):
        """
        更新 Actor 网络。

        参数:
            sample: 包含用于更新网络的数据批次。

        返回:
            policy_loss: 策略损失值。
            dist_entropy: 动作熵。
            actor_grad_norm: Actor 更新的梯度范数。
            imp_weights: 重要性采样权重。
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(
            old_action_log_probs_batch,
        ).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(
            active_masks_batch,
        ).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # 单次前向传播评估所有步骤
        action_log_probs, dist_entropy, _ = (
            self.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
            )
        )

        # 计算重要性权重并更新 Actor
        imp_weights = getattr(
            torch, self.action_aggregation,
        )(
            torch.exp(
                action_log_probs
                - old_action_log_probs_batch,
            ),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(
                imp_weights,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(
                    factor_batch * torch.min(surr1, surr2),
                    dim=-1,
                    keepdim=True,
                )
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2),
                dim=-1,
                keepdim=True,
            ).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        # 加入熵正则项后反向传播
        (
            policy_loss - dist_entropy * self.entropy_coef
        ).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
            )
        else:
            actor_grad_norm = get_grad_norm(
                self.actor.parameters(),
            )

        self.actor_optimizer.step()

        return (
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            imp_weights,
        )

    def train(
        self,
        actor_buffer,
        advantages,
        state_type,
    ):
        """
        使用小批量梯度下降执行一次训练更新。

        参数:
            actor_buffer: 包含 Actor 训练数据的缓冲区。
            advantages: 优势值数组。
            state_type: 状态类型。

        返回:
            train_info: 包含训练更新信息的字典。
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(
            actor_buffer.active_masks[:-1] == 0.0,
        ):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[
                actor_buffer.active_masks[:-1] == 0.0
            ] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (
                (advantages - mean_advantages)
                / (std_advantages + 1e-5)
            )

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = (
                    actor_buffer
                    .recurrent_generator_actor(
                        advantages,
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                )
            elif self.use_naive_recurrent_policy:
                data_generator = (
                    actor_buffer
                    .naive_recurrent_generator_actor(
                        advantages,
                        self.actor_num_mini_batch,
                    )
                )
            else:
                data_generator = (
                    actor_buffer
                    .feed_forward_generator_actor(
                        advantages,
                        self.actor_num_mini_batch,
                    )
                )

            for sample in data_generator:
                (
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    imp_weights,
                ) = self.update(sample)

                train_info["policy_loss"] += (
                    policy_loss.item()
                )
                train_info["dist_entropy"] += (
                    dist_entropy.item()
                )
                train_info["actor_grad_norm"] += (
                    actor_grad_norm
                )
                train_info["ratio"] += imp_weights.mean()

        num_updates = (
            self.ppo_epoch * self.actor_num_mini_batch
        )

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
