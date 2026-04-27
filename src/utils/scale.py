"""
在线值缩放模块。

提供基于百分位的运行缩放估计器，用于 Q 值归一化。
"""

import torch
import torch.nn as nn


class RunningScale(nn.Module):
    """
    基于百分位的在线缩放估计器。

    用第 5 和第 95 百分位的差值作为缩放因子，
    通过指数移动平均平滑更新。
    """

    def __init__(
        self,
        tau: float,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        初始化 RunningScale。

        参数:
            tau: 指数移动平均系数，越大更新越快。
            device: 张量所在设备。
        """
        super().__init__()
        self.tau = tau
        self.register_buffer(
            "_value",
            torch.ones(1, device=device),
        )
        self.register_buffer(
            "_percentiles",
            torch.tensor([5, 95], dtype=torch.float32,
                         device=device),
        )

    @property
    def value(self) -> float:
        """返回当前缩放因子的标量值。"""
        return self._value.cpu().item()

    def update(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        用新数据更新缩放因子。

        参数:
            x: 输入张量，至少为二维。
        """
        percentiles = self._percentile(x.detach())
        new_value = torch.clamp(
            percentiles[1] - percentiles[0], min=1.0,
        )
        self._value.data.lerp_(new_value, self.tau)

    def forward(
        self,
        x: torch.Tensor,
        update: bool = False,
    ) -> torch.Tensor:
        """
        对输入做缩放归一化。

        参数:
            x: 输入张量。
            update: 是否同时更新缩放因子。

        返回:
            缩放后的张量。
        """
        if update:
            self.update(x)
        return x / self._value

    def _percentile(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算输入的百分位值。

        参数:
            x: 输入张量，沿第 0 维排序计算。

        返回:
            形状为 (2, *x.shape[1:]) 的百分位张量。
        """
        x_dtype = x.dtype
        x_shape = x.shape
        x = x.flatten(1, x.ndim - 1)
        in_sorted = torch.sort(x, dim=0).values
        n = x.shape[0]
        positions = self._percentiles * (n - 1) / 100
        floored = torch.floor(positions).long()
        ceiled = torch.where(
            floored + 1 > n - 1,
            torch.tensor(n - 1, device=x.device),
            floored + 1,
        )
        weight_ceiled = (
            positions - floored.float()
        ).unsqueeze(1)
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored] * weight_floored
        d1 = in_sorted[ceiled] * weight_ceiled
        return (d0 + d1).reshape(
            -1, *x_shape[1:]
        ).to(x_dtype)

    def __repr__(self) -> str:
        """返回可读的字符串表示。"""
        return f"RunningScale(S: {self.value:.4f})"
