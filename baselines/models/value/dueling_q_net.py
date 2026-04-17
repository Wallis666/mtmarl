"""Dueling Q 网络模块。"""

import torch
import torch.nn as nn

from baselines.models.base.plain_cnn import PlainCNN
from baselines.models.base.plain_mlp import PlainMLP
from baselines.utils.env import get_shape_from_obs_space


class DuelingQNet(nn.Module):
    """Dueling Q 网络，适用于离散动作空间。"""

    def __init__(
        self,
        args,
        obs_space,
        output_dim,
        device=torch.device("cpu"),
    ):
        """
        初始化 DuelingQNet 模型。

        参数:
            args: 包含模型相关信息的参数字典。
            obs_space: 观测空间。
            output_dim: 输出维度。
            device: 运行设备。
        """
        super().__init__()
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        base_hidden_sizes = (
            args["base_hidden_sizes"]
        )
        base_activation_func = (
            args["base_activation_func"]
        )
        dueling_v_hidden_sizes = (
            args["dueling_v_hidden_sizes"]
        )
        dueling_v_activation_func = (
            args["dueling_v_activation_func"]
        )
        dueling_a_hidden_sizes = (
            args["dueling_a_hidden_sizes"]
        )
        dueling_a_activation_func = (
            args["dueling_a_activation_func"]
        )

        obs_shape = get_shape_from_obs_space(obs_space)

        # 特征提取器
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape,
                base_hidden_sizes[0],
                base_activation_func,
            )
            feature_dim = base_hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]

        # 基础网络
        base_sizes = (
            [feature_dim] + list(base_hidden_sizes)
        )
        self.base = PlainMLP(
            base_sizes,
            base_activation_func,
            base_activation_func,
        )

        # Dueling V 分支
        dueling_v_sizes = (
            [base_hidden_sizes[-1]]
            + list(dueling_v_hidden_sizes)
            + [1]
        )
        self.dueling_v = PlainMLP(
            dueling_v_sizes,
            dueling_v_activation_func,
        )

        # Dueling A 分支
        dueling_a_sizes = (
            [base_hidden_sizes[-1]]
            + list(dueling_a_hidden_sizes)
            + [output_dim]
        )
        self.dueling_a = PlainMLP(
            dueling_a_sizes,
            dueling_a_activation_func,
        )

        self.to(device)

    def forward(
        self,
        obs,
    ):
        """
        前向传播，计算 Q 值。

        参数:
            obs: 观测输入。

        返回:
            Dueling Q 值预测。
        """
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        x = self.base(x)
        v = self.dueling_v(x)
        a = self.dueling_a(x)
        return a - a.mean(dim=-1, keepdim=True) + v
