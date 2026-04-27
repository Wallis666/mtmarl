"""
数学工具模块。

提供 symlog/symexp 变换、Two-Hot 编码与解码、
以及分布式回归损失等数值处理工具。
"""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    对称对数变换：sign(x) * log(1 + |x|)。

    参数:
        x: 输入张量。

    返回:
        变换后的张量。
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    对称指数变换，symlog 的逆变换。

    参数:
        x: 输入张量。

    返回:
        变换后的张量。
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TwoHotProcessor:
    """
    Two-Hot 编码处理器。

    将连续标量转换为离散分布表示（Two-Hot 编码），
    并提供反向解码和分布式回归损失。支持三种模式：

    - num_bins == 0：直接透传，使用 MSE 损失。
    - num_bins == 1：仅做 symlog/symexp 变换。
    - num_bins > 1：完整 Two-Hot 编码。
    """

    def __init__(
        self,
        num_bins: int,
        vmin: float,
        vmax: float,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        初始化 TwoHotProcessor。

        参数:
            num_bins: 离散化的 bin 数量。
            vmin: symlog 空间中的最小值。
            vmax: symlog 空间中的最大值。
            device: 张量所在设备。
        """
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        if num_bins > 1:
            self.bin_size = (vmax - vmin) / (num_bins - 1)
            self.bins = torch.linspace(
                vmin, vmax, num_bins, device=device,
            )
        else:
            self.bin_size = 0.0
            self.bins = None

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        将标量编码为 Two-Hot 向量。

        参数:
            x: 输入标量张量，形状为 (batch_size, 1)。

        返回:
            编码后的张量。num_bins > 1 时形状为
            (batch_size, num_bins)，否则形状不变。
        """
        if self.num_bins == 0:
            return x
        if self.num_bins == 1:
            return symlog(x)
        x_log = torch.clamp(
            symlog(x), self.vmin, self.vmax,
        ).squeeze(1)
        bin_idx = torch.floor(
            (x_log - self.vmin) / self.bin_size,
        ).long()
        bin_offset = (
            (x_log - self.vmin) / self.bin_size
            - bin_idx.float()
        ).unsqueeze(-1)
        two_hot = torch.zeros(
            x.size(0), self.num_bins,
            device=x.device, dtype=x.dtype,
        )
        next_bin = (bin_idx + 1) % self.num_bins
        two_hot.scatter_(
            1, bin_idx.unsqueeze(1), 1 - bin_offset,
        )
        two_hot.scatter_(
            1, next_bin.unsqueeze(1), bin_offset,
        )
        return two_hot

    def decode(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 logits 解码为标量。

        参数:
            logits: 输入张量。num_bins > 1 时形状为
                (*, num_bins)，否则为 (*, 1)。

        返回:
            解码后的标量张量，形状为 (*, 1)。
        """
        if self.num_bins == 0:
            return logits
        if self.num_bins == 1:
            return symexp(logits)
        probs = F.softmax(logits, dim=-1)
        weighted = torch.sum(
            probs * self.bins, dim=-1, keepdim=True,
        )
        return symexp(weighted)

    def loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算分布式回归损失。

        参数:
            logits: 模型输出。num_bins > 1 时形状为
                (*, num_bins)。
            target: 目标标量，形状为 (*, 1)。

        返回:
            损失张量，形状为 (*, 1)。
        """
        if self.num_bins == 0:
            return F.mse_loss(logits, target, reduction="none")
        if self.num_bins == 1:
            return F.mse_loss(
                symexp(logits), target, reduction="none",
            )
        log_pred = F.log_softmax(logits, dim=-1)
        target_encoded = self.encode(target)
        return -(target_encoded * log_pred).sum(
            dim=-1, keepdim=True,
        )
