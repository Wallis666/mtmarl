"""简单 MLP 模块。"""

import torch.nn as nn

from baselines.utils.model import get_active_func


class PlainMLP(nn.Module):
    """简单多层感知机。"""

    def __init__(
        self,
        sizes,
        activation_func,
        final_activation_func="identity",
    ):
        """
        初始化 PlainMLP。

        参数:
            sizes: 各层大小列表。
            activation_func: 激活函数名称。
            final_activation_func: 最后一层的激活函数。
        """
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = (
                activation_func
                if j < len(sizes) - 2
                else final_activation_func
            )
            layers += [
                nn.Linear(sizes[j], sizes[j + 1]),
                get_active_func(act),
            ]
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        """
        前向传播。

        参数:
            x: 输入张量。

        返回:
            MLP 输出。
        """
        return self.mlp(x)
