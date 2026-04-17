"""HATRPO 算法模块。"""

import numpy as np
import torch

from baselines.algos.actors.on_policy_base import OnPolicyBase
from baselines.models.policy.stochastic_policy import (
    StochasticPolicy,
)
from baselines.utils.env import check
from baselines.utils.trpo import (
    conjugate_gradient,
    fisher_vector_product,
    flat_grad,
    flat_params,
    kl_divergence,
    update_model,
)


class HATRPO(OnPolicyBase):
    """HATRPO 算法。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 HATRPO 算法。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间（不支持 MultiDiscrete）。
            device: 用于张量运算的设备。
        """
        assert (
            act_space.__class__.__name__
            != "MultiDiscrete"
        ), "HATRPO 仅支持连续和离散动作空间。"
        super(HATRPO, self).__init__(
            args, obs_space, act_space, device,
        )

        self.kl_threshold = args["kl_threshold"]
        self.ls_step = args["ls_step"]
        self.accept_ratio = args["accept_ratio"]
        self.backtrack_coeff = args["backtrack_coeff"]

    def update(
        self,
        sample,
    ):
        """
        更新 Actor 网络。

        参数:
            sample: 包含用于更新网络的数据批次。

        返回:
            kl: 新旧策略之间的 KL 散度。
            loss_improve: 损失改进量。
            expected_improve: 预期损失改进量。
            dist_entropy: 动作熵。
            ratio: 新旧策略之间的比率。
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

        # 计算比率并更新 Actor
        ratio = getattr(
            torch, self.action_aggregation,
        )(
            torch.exp(
                action_log_probs
                - old_action_log_probs_batch,
            ),
            dim=-1,
            keepdim=True,
        )
        if self.use_policy_active_masks:
            loss = (
                torch.sum(
                    ratio * factor_batch * adv_targ,
                    dim=-1,
                    keepdim=True,
                )
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            loss = torch.sum(
                ratio * factor_batch * adv_targ,
                dim=-1,
                keepdim=True,
            ).mean()

        loss_grad = torch.autograd.grad(
            loss,
            self.actor.parameters(),
            allow_unused=True,
        )
        loss_grad = flat_grad(loss_grad)

        step_dir = conjugate_gradient(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            loss_grad.data,
            nsteps=10,
            device=self.device,
        )

        loss = loss.data.cpu().numpy()

        params = flat_params(self.actor)
        fvp = fisher_vector_product(
            self.actor,
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            step_dir,
        )
        shs = 0.5 * (step_dir * fvp).sum(
            0, keepdim=True,
        )
        step_size = 1 / torch.sqrt(
            shs / self.kl_threshold,
        )[0]
        full_step = step_size * step_dir

        old_actor = StochasticPolicy(
            self.args,
            self.obs_space,
            self.act_space,
            self.device,
        )
        update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(
            0, keepdim=True,
        )
        expected_improve = (
            expected_improve.data.cpu().numpy()
        )

        # 回溯线搜索
        flag = False
        fraction = 1
        for i in range(self.ls_step):
            new_params = params + fraction * full_step
            update_model(self.actor, new_params)
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

            ratio = getattr(
                torch, self.action_aggregation,
            )(
                torch.exp(
                    action_log_probs
                    - old_action_log_probs_batch,
                ),
                dim=-1,
                keepdim=True,
            )
            if self.use_policy_active_masks:
                new_loss = (
                    torch.sum(
                        ratio
                        * factor_batch
                        * adv_targ,
                        dim=-1,
                        keepdim=True,
                    )
                    * active_masks_batch
                ).sum() / active_masks_batch.sum()
            else:
                new_loss = torch.sum(
                    ratio
                    * factor_batch
                    * adv_targ,
                    dim=-1,
                    keepdim=True,
                ).mean()

            new_loss = new_loss.data.cpu().numpy()
            loss_improve = new_loss - loss

            kl = kl_divergence(
                obs_batch,
                rnn_states_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                new_actor=self.actor,
                old_actor=old_actor,
            )
            kl = kl.mean()

            if (
                kl < self.kl_threshold
                and (loss_improve / expected_improve)
                > self.accept_ratio
                and loss_improve.item() > 0
            ):
                flag = True
                break
            expected_improve *= self.backtrack_coeff
            fraction *= self.backtrack_coeff

        if not flag:
            params = flat_params(old_actor)
            update_model(self.actor, params)
            print("策略更新未能改进替代目标函数")

        return (
            kl,
            loss_improve,
            expected_improve,
            dist_entropy,
            ratio,
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
        train_info["kl"] = 0
        train_info["dist_entropy"] = 0
        train_info["loss_improve"] = 0
        train_info["expected_improve"] = 0
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

        if self.use_recurrent_policy:
            data_generator = (
                actor_buffer
                .recurrent_generator_actor(
                    advantages,
                    1,
                    self.data_chunk_length,
                )
            )
        elif self.use_naive_recurrent_policy:
            data_generator = (
                actor_buffer
                .naive_recurrent_generator_actor(
                    advantages, 1,
                )
            )
        else:
            data_generator = (
                actor_buffer
                .feed_forward_generator_actor(
                    advantages, 1,
                )
            )

        for sample in data_generator:
            (
                kl,
                loss_improve,
                expected_improve,
                dist_entropy,
                imp_weights,
            ) = self.update(sample)

            train_info["kl"] += kl
            train_info["loss_improve"] += (
                loss_improve.item()
            )
            train_info["expected_improve"] += (
                expected_improve
            )
            train_info["dist_entropy"] += (
                dist_entropy.item()
            )
            train_info["ratio"] += imp_weights.mean()

        num_updates = 1

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
