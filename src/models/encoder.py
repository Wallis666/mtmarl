"""
观测编码器模块。

提供状态编码器和图像编码器的统一构建接口，
将原始观测映射到 latent 空间表示。
"""

import torch
import torch.nn as nn

from src.models.base.mlp import SimNorm, mlp
from src.models.base.cnn import cnn


class StateEncoder(nn.Module):
    """
    状态观测编码器。

    将向量形式的状态观测编码为 latent 表示，
    使用 MLP + SimNorm 输出归一化。
    """

    def __init__(
        self,
        obs_dim: int,
        enc_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        simnorm_dim: int = 8,
    ) -> None:
        """
        初始化 StateEncoder。

        参数:
            obs_dim: 观测向量维度。
            enc_dim: 编码器隐藏层维度。
            latent_dim: 输出 latent 维度。
            num_layers: 隐藏层数量。
            simnorm_dim: SimNorm 分组大小。
        """
        super().__init__()
        self.net = mlp(
            in_dim=obs_dim,
            mlp_dims=[enc_dim] * max(num_layers - 1, 1),
            out_dim=latent_dim,
            act=SimNorm(simnorm_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码状态观测。

        参数:
            obs: 状态观测，形状为 (*, obs_dim)。

        返回:
            latent 表示，形状为 (*, latent_dim)。
        """
        return self.net(obs)


class ImageEncoder(nn.Module):
    """
    图像观测编码器。

    将 64x64 的图像观测编码为 latent 表示，
    使用 4 层 CNN + SimNorm 输出归一化。
    """

    def __init__(
        self,
        in_shape: tuple[int, ...],
        num_channels: int,
        simnorm_dim: int = 8,
    ) -> None:
        """
        初始化 ImageEncoder。

        参数:
            in_shape: 图像形状 (C, H, W)，要求 W == 64。
            num_channels: 卷积层通道数。
            simnorm_dim: SimNorm 分组大小。
        """
        super().__init__()
        self.net = cnn(
            in_shape=in_shape,
            num_channels=num_channels,
            act=SimNorm(simnorm_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码图像观测。

        参数:
            obs: 图像观测，形状为 (N, C, H, W)。
                支持带时间维的 (T, N, C, H, W)，
                此时逐帧编码。

        返回:
            latent 表示，形状为 (N, latent_dim)
            或 (T, N, latent_dim)。
        """
        if obs.ndim == 5:
            return torch.stack(
                [self.net(o) for o in obs]
            )
        return self.net(obs)


def encoder(
    obs_type: str,
    obs_dim: int | tuple[int, ...],
    latent_dim: int,
    enc_dim: int = 256,
    num_layers: int = 2,
    num_channels: int = 32,
    simnorm_dim: int = 8,
) -> nn.Module:
    """
    根据观测类型构建对应的编码器。

    参数:
        obs_type: 观测类型，"state" 或 "rgb"。
        obs_dim: 观测维度。state 时为整数，
            rgb 时为 (C, H, W) 元组。
        latent_dim: 输出 latent 维度。
        enc_dim: 状态编码器隐藏层维度。
        num_layers: 状态编码器隐藏层数量。
        num_channels: 图像编码器卷积通道数。
        simnorm_dim: SimNorm 分组大小。

    返回:
        构建好的编码器模块。

    异常:
        ValueError: 当 obs_type 不是
            "state" 或 "rgb" 时抛出。
    """
    if obs_type == "state":
        assert isinstance(obs_dim, int), (
            f"state 编码器要求 obs_dim 为整数，"
            f"得到 {type(obs_dim)}"
        )
        return StateEncoder(
            obs_dim=obs_dim,
            enc_dim=enc_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            simnorm_dim=simnorm_dim,
        )
    elif obs_type == "rgb":
        assert isinstance(obs_dim, tuple), (
            f"rgb 编码器要求 obs_dim 为元组，"
            f"得到 {type(obs_dim)}"
        )
        return ImageEncoder(
            in_shape=obs_dim,
            num_channels=num_channels,
            simnorm_dim=simnorm_dim,
        )
    else:
        raise ValueError(
            f"不支持的观测类型: {obs_type}，"
            f"仅支持 'state' 和 'rgb'"
        )
