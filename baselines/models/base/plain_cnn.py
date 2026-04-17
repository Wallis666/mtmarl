"""简单 CNN 模块。"""

import torch.nn as nn

from baselines.models.base.flatten import Flatten
from baselines.utils.model import get_active_func


class PlainCNN(nn.Module):
    """简单卷积神经网络。"""

    def __init__(
        self,
        obs_shape,
        hidden_size,
        activation_func,
        kernel_size=3,
        stride=1,
    ):
        """
        初始化 PlainCNN。

        参数:
            obs_shape: 观测形状。
            hidden_size: 隐藏层大小。
            activation_func: 激活函数名称。
            kernel_size: 卷积核大小。
            stride: 步长。
        """
        super().__init__()
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]
        layers = [
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=hidden_size // 4,
                kernel_size=kernel_size,
                stride=stride,
            ),
            get_active_func(activation_func),
            Flatten(),
            nn.Linear(
                hidden_size
                // 4
                * (input_width - kernel_size + stride)
                * (
                    input_height
                    - kernel_size
                    + stride
                ),
                hidden_size,
            ),
            get_active_func(activation_func),
        ]
        self.cnn = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        """
        前向传播。

        参数:
            x: 输入张量。

        返回:
            CNN 输出。
        """
        x = x / 255.0
        x = self.cnn(x)
        return x
