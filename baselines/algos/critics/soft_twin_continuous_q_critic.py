"""Soft 双 Q 网络连续动作空间 Critic 模块。"""

import numpy as np
import torch
import torch.nn.functional as F

from baselines.algos.critics.twin_continuous_q_critic import (
    TwinContinuousQCritic,
)
from baselines.utils.env import check


class SoftTwinContinuousQCritic(TwinContinuousQCritic):
    """学习两个 Soft Q 函数的 Critic，支持连续和离散动作空间。"""

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """
        初始化 SoftTwinContinuousQCritic。

        参数:
            args: 算法参数字典。
            share_obs_space: 共享观测空间。
            act_space: 动作空间。
            num_agents: 智能体数量。
            state_type: 状态类型。
            device: 用于张量运算的设备。
        """
        super(
            SoftTwinContinuousQCritic, self,
        ).__init__(
            args,
            share_obs_space,
            act_space,
            num_agents,
            state_type,
            device,
        )

        self.tpdv_a = dict(
            dtype=torch.int64, device=device,
        )
        self.auto_alpha = args["auto_alpha"]
        if self.auto_alpha:
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=device,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=args["alpha_lr"],
            )
            self.alpha = torch.exp(
                self.log_alpha.detach(),
            )
        else:
            self.alpha = args["alpha"]
        self.use_policy_active_masks = (
            args["use_policy_active_masks"]
        )
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

    def update_alpha(
        self,
        logp_actions,
        target_entropy,
    ):
        """自动调节温度参数 alpha。"""
        log_prob = (
            torch.sum(
                torch.cat(logp_actions, dim=-1),
                dim=-1,
                keepdim=True,
            )
            .detach()
            .to(**self.tpdv)
            + target_entropy
        )
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(
            self.log_alpha.detach(),
        )

    def get_values(
        self,
        share_obs,
        actions,
    ):
        """获取 Soft Q 值。"""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return torch.min(
            self.critic(share_obs, actions),
            self.critic2(share_obs, actions),
        )

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        valid_transition,
        term,
        next_share_obs,
        next_actions,
        next_logp_actions,
        gamma,
        value_normalizer=None,
    ):
        """
        训练 Critic。

        参数:
            share_obs: 共享观测。
            actions: 动作。
            reward: 奖励。
            done: 终止标志。
            valid_transition: 有效转移掩码。
            term: 截断标志。
            next_share_obs: 下一步共享观测。
            next_actions: 下一步动作。
            next_logp_actions: 下一步动作的对数概率。
            gamma: 折扣因子。
            value_normalizer: 奖励归一化器。
        """
        assert (
            share_obs.__class__.__name__ == "ndarray"
        )
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert (
            next_share_obs.__class__.__name__
            == "ndarray"
        )
        assert gamma.__class__.__name__ == "ndarray"

        share_obs = check(share_obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions = check(actions).to(**self.tpdv)
            actions = torch.cat(
                [
                    actions[i]
                    for i in range(actions.shape[0])
                ],
                dim=-1,
            )
        else:
            actions = check(actions).to(**self.tpdv_a)
            one_hot_actions = []
            for agent_id in range(len(actions)):
                if self.action_type == "MultiDiscrete":
                    action_dims = (
                        self.act_space[agent_id].nvec
                    )
                    one_hot_action = []
                    for dim in range(len(action_dims)):
                        one_hot = F.one_hot(
                            actions[agent_id, :, dim],
                            num_classes=action_dims[dim],
                        )
                        one_hot_action.append(one_hot)
                    one_hot_action = torch.cat(
                        one_hot_action, dim=-1,
                    )
                else:
                    one_hot_action = F.one_hot(
                        actions[agent_id],
                        num_classes=(
                            self.act_space[agent_id].n
                        ),
                    )
                one_hot_actions.append(one_hot_action)
            actions = torch.squeeze(
                torch.cat(one_hot_actions, dim=-1),
                dim=1,
            ).to(**self.tpdv_a)
        if self.state_type == "FP":
            actions = torch.tile(
                actions, (self.num_agents, 1),
            )
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        valid_transition = check(
            np.concatenate(valid_transition, axis=0),
        ).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(
            **self.tpdv,
        )
        if self.action_type == "Box":
            next_actions = torch.cat(
                next_actions, dim=-1,
            ).to(**self.tpdv)
        else:
            next_actions = torch.cat(
                next_actions, dim=-1,
            ).to(**self.tpdv_a)
        next_logp_actions = torch.sum(
            torch.cat(next_logp_actions, dim=-1),
            dim=-1,
            keepdim=True,
        ).to(**self.tpdv)
        if self.state_type == "FP":
            next_actions = torch.tile(
                next_actions, (self.num_agents, 1),
            )
            next_logp_actions = torch.tile(
                next_logp_actions,
                (self.num_agents, 1),
            )
        next_q_values1 = self.target_critic(
            next_share_obs, next_actions,
        )
        next_q_values2 = self.target_critic2(
            next_share_obs, next_actions,
        )
        next_q_values = torch.min(
            next_q_values1, next_q_values2,
        )
        if self.use_proper_time_limits:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check(
                        value_normalizer.denormalize(
                            next_q_values,
                        ),
                    ).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - term)
                value_normalizer.update(q_targets)
                q_targets = check(
                    value_normalizer.normalize(
                        q_targets,
                    ),
                ).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values
                    - self.alpha * next_logp_actions
                ) * (1 - term)
        else:
            if value_normalizer is not None:
                q_targets = reward + gamma * (
                    check(
                        value_normalizer.denormalize(
                            next_q_values,
                        ),
                    ).to(**self.tpdv)
                    - self.alpha * next_logp_actions
                ) * (1 - done)
                value_normalizer.update(q_targets)
                q_targets = check(
                    value_normalizer.normalize(
                        q_targets,
                    ),
                ).to(**self.tpdv)
            else:
                q_targets = reward + gamma * (
                    next_q_values
                    - self.alpha * next_logp_actions
                ) * (1 - done)
        if self.use_huber_loss:
            if (
                self.state_type == "FP"
                and self.use_policy_active_masks
            ):
                critic_loss1 = (
                    torch.sum(
                        F.huber_loss(
                            self.critic(
                                share_obs, actions,
                            ),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
                critic_loss2 = (
                    torch.mean(
                        F.huber_loss(
                            self.critic2(
                                share_obs, actions,
                            ),
                            q_targets,
                            delta=self.huber_delta,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss1 = torch.mean(
                    F.huber_loss(
                        self.critic(
                            share_obs, actions,
                        ),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
                critic_loss2 = torch.mean(
                    F.huber_loss(
                        self.critic2(
                            share_obs, actions,
                        ),
                        q_targets,
                        delta=self.huber_delta,
                    )
                )
        else:
            if (
                self.state_type == "FP"
                and self.use_policy_active_masks
            ):
                critic_loss1 = (
                    torch.sum(
                        F.mse_loss(
                            self.critic(
                                share_obs, actions,
                            ),
                            q_targets,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
                critic_loss2 = (
                    torch.sum(
                        F.mse_loss(
                            self.critic2(
                                share_obs, actions,
                            ),
                            q_targets,
                        )
                        * valid_transition
                    )
                    / valid_transition.sum()
                )
            else:
                critic_loss1 = torch.mean(
                    F.mse_loss(
                        self.critic(
                            share_obs, actions,
                        ),
                        q_targets,
                    )
                )
                critic_loss2 = torch.mean(
                    F.mse_loss(
                        self.critic2(
                            share_obs, actions,
                        ),
                        q_targets,
                    )
                )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
