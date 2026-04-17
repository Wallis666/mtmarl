"""随机策略模块，根据观测输出动作，支持循环神经网络。"""

import torch
import torch.nn as nn

from baselines.models.base.act import ACTLayer
from baselines.models.base.cnn import CNNBase
from baselines.models.base.mlp import MLPBase
from baselines.models.base.rnn import RNNLayer
from baselines.utils.env import check, get_shape_from_obs_space


class StochasticPolicy(nn.Module):
    """随机策略模型，根据观测输出动作。"""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        """初始化 StochasticPolicy 模型。

        参数:
            args: 包含模型相关信息的字典。
            obs_space: 观测空间。
            action_space: 动作空间。
            device: 指定运行设备（cpu/gpu）。
        """
        super(StochasticPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = (
            args["initialization_method"]
        )
        self.use_policy_active_masks = (
            args["use_policy_active_masks"]
        )
        self.use_naive_recurrent_policy = (
            args["use_naive_recurrent_policy"]
        )
        self.use_recurrent_policy = (
            args["use_recurrent_policy"]
        )
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(
            dtype=torch.float32,
            device=device,
        )

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if (
            self.use_naive_recurrent_policy
            or self.use_recurrent_policy
        ):
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        """根据给定输入计算动作。

        参数:
            obs: 输入网络的观测数据。
            rnn_states: 若使用 RNN 网络，
                则为 RNN 的隐藏状态。
            masks: 掩码张量，表示是否应将隐藏状态
                重新初始化为零。
            available_actions: 表示智能体可用的动作
                （若为 None 则所有动作可用）。
            deterministic: 是否从动作分布中采样，
                而非返回众数。

        返回:
            actions: 要执行的动作。
            action_log_probs: 所选动作的对数概率。
            rnn_states: 更新后的 RNN 隐藏状态。
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = (
                check(available_actions).to(**self.tpdv)
            )

        actor_features = self.base(obs)

        if (
            self.use_naive_recurrent_policy
            or self.use_recurrent_policy
        ):
            actor_features, rnn_states = self.rnn(
                actor_features,
                rnn_states,
                masks,
            )

        actions, action_log_probs = self.act(
            actor_features,
            available_actions,
            deterministic,
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """计算动作对数概率、分布熵和动作分布。

        参数:
            obs: 输入网络的观测数据。
            rnn_states: 若使用 RNN 网络，
                则为 RNN 的隐藏状态。
            action: 需要评估熵和对数概率的动作。
            masks: 掩码张量，表示是否应将隐藏状态
                重新初始化为零。
            available_actions: 表示智能体可用的动作
                （若为 None 则所有动作可用）。
            active_masks: 表示智能体是否存活。

        返回:
            action_log_probs: 输入动作的对数概率。
            dist_entropy: 给定输入的动作分布熵。
            action_distribution: 动作分布。
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = (
                check(available_actions).to(**self.tpdv)
            )

        if active_masks is not None:
            active_masks = (
                check(active_masks).to(**self.tpdv)
            )

        actor_features = self.base(obs)

        if (
            self.use_naive_recurrent_policy
            or self.use_recurrent_policy
        ):
            actor_features, rnn_states = self.rnn(
                actor_features,
                rnn_states,
                masks,
            )

        active = (
            active_masks
            if self.use_policy_active_masks
            else None
        )
        action_log_probs, dist_entropy, action_distribution = (
            self.act.evaluate_actions(
                actor_features,
                action,
                available_actions,
                active_masks=active,
            )
        )

        return (
            action_log_probs,
            dist_entropy,
            action_distribution,
        )
