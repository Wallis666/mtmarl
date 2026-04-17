"""展平模块。"""

from torch import nn


class Flatten(nn.Module):
    """将输入张量展平为二维。"""

    def forward(
        self,
        x,
    ):
        """
        前向传播。

        参数:
            x: 输入张量。

        返回:
            展平后的张量。
        """
        return x.view(x.size(0), -1)
