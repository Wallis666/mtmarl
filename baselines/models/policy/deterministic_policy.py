"""确定性策略网络模块，用于连续动作空间。"""

import torch
import torch.nn as nn

from baselines.models.base.plain_cnn import PlainCNN
from baselines.models.base.plain_mlp import PlainMLP
from baselines.utils.env import get_shape_from_obs_space


class DeterministicPolicy(nn.Module):
    """连续动作空间的确定性策略网络。"""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        """初始化确定性策略模型。

        Args:
            args: 包含模型相关信息的字典。
            obs_space: 观测空间。
            action_space: 动作空间。
            device: 指定运行设备（cpu/gpu）。
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape,
                hidden_sizes[0],
                activation_func,
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]
        act_dim = action_space.shape[0]
        pi_sizes = (
            [feature_dim] + list(hidden_sizes) + [act_dim]
        )
        self.pi = PlainMLP(
            pi_sizes,
            activation_func,
            final_activation_func,
        )
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(
            **self.tpdv
        )
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
        self.to(device)

    def forward(
        self,
        obs,
    ):
        """前向传播，返回缩放到动作空间范围内的输出。"""
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        x = self.pi(x)
        x = self.scale * x + self.mean
        return x
