"""随机多层感知机策略模块，仅使用 MLP 网络根据观测输出动作。"""

import torch
import torch.nn as nn

from baselines.models.base.act import ACTLayer
from baselines.models.base.cnn import CNNBase
from baselines.models.base.mlp import MLPBase
from baselines.utils.env import check, get_shape_from_obs_space


class StochasticMlpPolicy(nn.Module):
    """仅使用 MLP 的随机策略模型，根据观测输出动作。"""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        """初始化 StochasticMlpPolicy 模型。

        参数:
            args: 包含模型相关信息的字典。
            obs_space: 观测空间。
            action_space: 动作空间。
            device: 指定运行设备（cpu/gpu）。
        """
        super(StochasticMlpPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = (
            args["initialization_method"]
        )

        self.tpdv = dict(
            dtype=torch.float32,
            device=device,
        )

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
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
        available_actions=None,
        stochastic=True,
    ):
        """根据给定输入计算动作。

        参数:
            obs: 输入网络的观测数据。
            available_actions: 表示智能体可用的动作
                （若为 None 则所有动作可用）。
            stochastic: 是否从动作分布中采样，
                而非返回众数。

        返回:
            actions: 要执行的动作。
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = (
                check(available_actions).to(**self.tpdv)
            )

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(
            actor_features,
            available_actions,
            deterministic,
        )
        return actions

    def get_logits(
        self,
        obs,
        available_actions=None,
    ):
        """根据给定输入获取动作 logits。

        参数:
            obs: 输入网络的数据。
            available_actions: 表示智能体可用的动作
                （若为 None 则所有动作可用）。

        返回:
            action_logits: 给定输入对应的动作 logits。
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = (
                check(available_actions).to(**self.tpdv)
            )

        actor_features = self.base(obs)

        return self.act.get_logits(
            actor_features,
            available_actions,
        )
