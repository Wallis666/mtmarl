"""卷积神经网络模块。"""

import torch.nn as nn

from baselines.models.base.flatten import Flatten
from baselines.utils.model import (
    get_active_func,
    get_init_method,
    init,
)


class CNNLayer(nn.Module):
    """卷积神经网络层。"""

    def __init__(
        self,
        obs_shape,
        hidden_sizes,
        initialization_method,
        activation_func,
        kernel_size=3,
        stride=1,
    ):
        super(CNNLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(
                m,
                init_method,
                lambda x: nn.init.constant_(x, 0),
                gain=gain,
            )

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        layers = [
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_sizes[0] // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    hidden_sizes[0]
                    // 2
                    * (input_width - kernel_size + stride)
                    * (input_height - kernel_size + stride),
                    hidden_sizes[0],
                )
            ),
            active_func,
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(
                    nn.Linear(
                        hidden_sizes[i - 1],
                        hidden_sizes[i],
                    )
                ),
                active_func,
            ]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。"""
        x = x / 255.0
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    """用于演员和评论家的卷积神经网络基础模块。"""

    def __init__(
        self,
        args,
        obs_shape,
    ):
        super(CNNBase, self).__init__()

        self.initialization_method = (
            args["initialization_method"]
        )
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        self.cnn = CNNLayer(
            obs_shape,
            self.hidden_sizes,
            self.initialization_method,
            self.activation_func,
        )

    def forward(self, x):
        """前向传播。"""
        x = self.cnn(x)
        return x
