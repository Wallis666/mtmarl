"""
卷积神经网络模块。

提供 TD-MPC2 风格的图像编码构建块，包含随机平移数据
增强、像素预处理，以及 4 层卷积编码器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftAug(nn.Module):
    """随机平移图像数据增强。"""

    def __init__(
        self,
        pad: int = 3,
    ) -> None:
        """
        初始化 ShiftAug。

        参数:
            pad: 填充像素数，平移范围为 [0, 2*pad]。
        """
        super().__init__()
        self.pad = pad
        self.padding = tuple([self.pad] * 4)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        对输入图像施加随机平移增强。

        参数:
            x: 输入图像，形状为 (N, C, H, W)，
                要求 H == W。

        返回:
            平移后的图像，形状不变。
        """
        x = x.float()
        n, _, h, w = x.size()
        assert h == w, f"要求输入为正方形图像，得到 {h}x{w}"
        x = F.pad(x, self.padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat(
            [arange, arange.transpose(1, 0)], dim=2,
        )
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype,
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False,
        )


class PixelPreprocess(nn.Module):
    """像素归一化层，将 [0, 255] 映射到 [-0.5, 0.5]。"""

    def __init__(
        self,
    ) -> None:
        """初始化 PixelPreprocess。"""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。"""
        return x.div(255.0).sub(0.5)


def cnn(
    in_shape: tuple[int, ...],
    num_channels: int,
    act: nn.Module | None = None,
) -> nn.Sequential:
    """
    创建 TD-MPC2 风格的卷积编码器。

    结构为 ShiftAug -> PixelPreprocess -> 4 层 Conv2d+ReLU
    -> Flatten，可选末尾激活（如 SimNorm）。
    要求输入图像宽度为 64。

    参数:
        in_shape: 输入形状 (C, H, W)，要求 W == 64。
        num_channels: 卷积层通道数。
        act: 末尾激活函数，为 None 时不添加。

    返回:
        构建好的 nn.Sequential 模型。
    """
    assert in_shape[-1] == 64, (
        f"要求输入图像宽度为 64，得到 {in_shape[-1]}"
    )
    layers: list[nn.Module] = [
        ShiftAug(),
        PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3, stride=2),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3, stride=1),
        nn.Flatten(),
    ]
    if act:
        layers.append(act)
    return nn.Sequential(*layers)
