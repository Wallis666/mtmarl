"""动作层模块，用于计算智能体的动作输出。"""

import torch
import torch.nn as nn

from baselines.models.base.distributions import (
    Categorical,
    DiagGaussian,
)


class ACTLayer(nn.Module):
    """多层感知机模块，用于计算动作。"""

    def __init__(
        self,
        action_space,
        inputs_dim,
        initialization_method,
        gain,
        args=None,
    ):
        """初始化动作层。

        参数:
            action_space: 动作空间。
            inputs_dim: 网络输入维度。
            initialization_method: 初始化方法。
            gain: 网络输出层的增益。
            args: 与网络相关的参数字典。
        """
        super(ACTLayer, self).__init__()
        self.action_type = action_space.__class__.__name__
        self.multidiscrete_action = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim,
                action_dim,
                initialization_method,
                gain,
            )
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                inputs_dim,
                action_dim,
                initialization_method,
                gain,
                args,
            )
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            action_dims = action_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(
                    Categorical(
                        inputs_dim,
                        action_dim,
                        initialization_method,
                        gain,
                    )
                )
            self.action_outs = nn.ModuleList(action_outs)

    def forward(
        self,
        x,
        available_actions=None,
        deterministic=False,
    ):
        """根据输入计算动作及其对数概率。

        参数:
            x: 网络输入张量。
            available_actions: 表示智能体可用动作的张量，
                若为 None 则所有动作均可用。
            deterministic: 是否返回确定性动作。

        返回:
            actions: 要执行的动作张量。
            action_log_probs: 所选动作的对数概率张量。
        """
        if self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_distribution = action_out(
                    x,
                    available_actions,
                )
                action = (
                    action_distribution.mode()
                    if deterministic
                    else action_distribution.sample()
                )
                action_log_prob = action_distribution.log_probs(
                    action,
                )
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(
                action_log_probs,
                dim=-1,
            ).sum(dim=-1, keepdim=True)
        else:
            action_distribution = self.action_out(
                x,
                available_actions,
            )
            actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
            action_log_probs = action_distribution.log_probs(
                actions,
            )

        return actions, action_log_probs

    def get_logits(
        self,
        x,
        available_actions=None,
    ):
        """根据输入获取动作的 logits。

        参数:
            x: 网络输入张量。
            available_actions: 表示智能体可用动作的张量，
                若为 None 则所有动作均可用。

        返回:
            action_logits: 给定输入对应的动作 logits 张量。
        """
        if self.multidiscrete_action:
            action_logits = []
            for action_out in self.action_outs:
                action_distribution = action_out(
                    x,
                    available_actions,
                )
                action_logits.append(
                    action_distribution.logits,
                )
        else:
            action_distribution = self.action_out(
                x,
                available_actions,
            )
            action_logits = action_distribution.logits

        return action_logits

    def evaluate_actions(
        self,
        x,
        action,
        available_actions=None,
        active_masks=None,
    ):
        """计算动作对数概率、分布熵和动作分布。

        参数:
            x: 网络输入张量。
            action: 需要评估熵和对数概率的动作张量。
            available_actions: 表示智能体可用动作的张量，
                若为 None 则所有动作均可用。
            active_masks: 表示智能体是否存活的掩码张量。

        返回:
            action_log_probs: 输入动作的对数概率张量。
            dist_entropy: 给定输入的动作分布熵张量。
            action_distribution: 动作分布对象。
        """
        if self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(
                self.action_outs,
                action,
            ):
                action_distribution = action_out(x)
                action_log_probs.append(
                    action_distribution.log_probs(
                        act.unsqueeze(-1),
                    )
                )
                if active_masks is not None:
                    dist_entropy.append(
                        (
                            action_distribution.entropy()
                            * active_masks
                        )
                        / active_masks.sum()
                    )
                else:
                    dist_entropy.append(
                        action_distribution.entropy()
                        / action_log_probs[-1].size(0)
                    )
            action_log_probs = torch.cat(
                action_log_probs,
                dim=-1,
            ).sum(dim=-1, keepdim=True)
            dist_entropy = (
                torch.cat(dist_entropy, dim=-1)
                .sum(dim=-1, keepdim=True)
                .mean()
            )
            return action_log_probs, dist_entropy, None
        else:
            action_distribution = self.action_out(
                x,
                available_actions,
            )
            action_log_probs = action_distribution.log_probs(
                action,
            )
            if active_masks is not None:
                if self.action_type == "Discrete":
                    dist_entropy = (
                        action_distribution.entropy()
                        * active_masks.squeeze(-1)
                    ).sum() / active_masks.sum()
                else:
                    dist_entropy = (
                        action_distribution.entropy()
                        * active_masks.squeeze(-1)
                    ).sum() / active_masks.sum()
            else:
                dist_entropy = (
                    action_distribution.entropy().mean()
                )

        return (
            action_log_probs,
            dist_entropy,
            action_distribution,
        )
