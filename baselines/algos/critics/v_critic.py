"""V Critic 模块。"""

import torch
import torch.nn as nn

from baselines.models.value.v_net import VNet
from baselines.utils.env import check
from baselines.utils.model import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)


class VCritic:
    """学习 V 函数的 Critic。"""

    def __init__(
        self,
        args,
        cent_obs_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 VCritic。

        参数:
            args: 算法参数字典。
            cent_obs_space: 中心化观测空间。
            device: 用于张量运算的设备。
        """
        self.args = args
        self.device = device
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = (
            args["critic_num_mini_batch"]
        )
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = (
            args["use_recurrent_policy"]
        )
        self.use_naive_recurrent_policy = (
            args["use_naive_recurrent_policy"]
        )
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = (
            args["use_clipped_value_loss"]
        )
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = (
            args["use_policy_active_masks"]
        )

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.share_obs_space = cent_obs_space

        self.critic = VNet(
            args, self.share_obs_space, self.device,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(
        self,
        episode,
        episodes,
    ):
        """
        衰减学习率。

        参数:
            episode: 当前训练回合数。
            episodes: 总训练回合数。
        """
        update_linear_schedule(
            self.critic_optimizer,
            episode,
            episodes,
            self.critic_lr,
        )

    def get_values(
        self,
        cent_obs,
        rnn_states_critic,
        masks,
    ):
        """
        获取价值函数预测。

        参数:
            cent_obs: Critic 的中心化观测输入。
            rnn_states_critic: Critic 的 RNN 状态。
            masks: 指示 RNN 状态是否需要重置的掩码。

        返回:
            values: 价值函数预测值。
            rnn_states_critic: 更新后的 RNN 状态。
        """
        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks,
        )
        return values, rnn_states_critic

    def cal_value_loss(
        self,
        values,
        value_preds_batch,
        return_batch,
        value_normalizer=None,
    ):
        """
        计算价值函数损失。

        参数:
            values: 价值函数预测值。
            value_preds_batch: 数据批次中的旧预测值。
            return_batch: 回报值。
            value_normalizer: 奖励归一化器。

        返回:
            value_loss: 价值函数损失。
        """
        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch
        ).clamp(-self.clip_param, self.clip_param)
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch)
                - value_pred_clipped
            )
            error_original = (
                value_normalizer.normalize(return_batch)
                - values
            )
        else:
            error_clipped = (
                return_batch - value_pred_clipped
            )
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(
                error_clipped, self.huber_delta,
            )
            value_loss_original = huber_loss(
                error_original, self.huber_delta,
            )
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(
                value_loss_original, value_loss_clipped,
            )
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(
        self,
        sample,
        value_normalizer=None,
    ):
        """
        更新 Critic 网络。

        参数:
            sample: 包含用于更新网络的数据批次。
            value_normalizer: 奖励归一化器。

        返回:
            value_loss: 价值函数损失。
            critic_grad_norm: Critic 更新的梯度范数。
        """
        (
            share_obs_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        value_preds_batch = check(
            value_preds_batch,
        ).to(**self.tpdv)
        return_batch = check(return_batch).to(
            **self.tpdv,
        )

        values, _ = self.get_values(
            share_obs_batch,
            rnn_states_critic_batch,
            masks_batch,
        )

        value_loss = self.cal_value_loss(
            values,
            value_preds_batch,
            return_batch,
            value_normalizer=value_normalizer,
        )

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )
        else:
            critic_grad_norm = get_grad_norm(
                self.critic.parameters(),
            )

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(
        self,
        critic_buffer,
        value_normalizer=None,
    ):
        """
        使用小批量梯度下降执行一次训练更新。

        参数:
            critic_buffer: 包含 Critic 训练数据的缓冲区。
            value_normalizer: 奖励归一化器。

        返回:
            train_info: 包含训练更新信息的字典。
        """
        train_info = {}
        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = (
                    critic_buffer
                    .recurrent_generator_critic(
                        self.critic_num_mini_batch,
                        self.data_chunk_length,
                    )
                )
            elif self.use_naive_recurrent_policy:
                data_generator = (
                    critic_buffer
                    .naive_recurrent_generator_critic(
                        self.critic_num_mini_batch,
                    )
                )
            else:
                data_generator = (
                    critic_buffer
                    .feed_forward_generator_critic(
                        self.critic_num_mini_batch,
                    )
                )

            for sample in data_generator:
                value_loss, critic_grad_norm = (
                    self.update(
                        sample,
                        value_normalizer=value_normalizer,
                    )
                )

                train_info["value_loss"] += (
                    value_loss.item()
                )
                train_info["critic_grad_norm"] += (
                    critic_grad_norm
                )

        num_updates = (
            self.critic_epoch
            * self.critic_num_mini_batch
        )

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """切换到训练模式。"""
        self.critic.train()

    def prep_rollout(self):
        """切换到评估模式。"""
        self.critic.eval()
