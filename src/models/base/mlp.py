"""
多层感知机模块。

提供 TD-MPC2 风格的 MLP 构建块，包含 LayerNorm + Mish 激活
的线性层，以及 Simplicial Normalization。
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Simplicial 归一化层。"""

    def __init__(
        self,
        dim: int,
    ) -> None:
        """
        初始化 SimNorm。

        参数:
            dim: 分组维度，对最后一维按此大小分组
                后做 softmax。
        """
        super().__init__()
        self.dim = dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。"""
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self) -> str:
        """返回可读的字符串表示。"""
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """带 LayerNorm 和激活函数的线性层。"""

    def __init__(
        self,
        *args: Any,
        dropout: float = 0.0,
        act: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 NormedLinear。

        参数:
            *args: 传递给 nn.Linear 的位置参数。
            dropout: Dropout 概率，为 0 时不使用。
            act: 激活函数，默认为 Mish。
            **kwargs: 传递给 nn.Linear 的关键字参数。
        """
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish()
        self.dropout = (
            nn.Dropout(dropout) if dropout else None
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：Linear -> Dropout -> LayerNorm -> Act。"""
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self) -> str:
        """返回可读的字符串表示。"""
        repr_dropout = (
            f", dropout={self.dropout.p}"
            if self.dropout
            else ""
        )
        return (
            f"NormedLinear("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
            f"{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(
    in_dim: int,
    mlp_dims: list[int] | int,
    out_dim: int,
    act: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    创建 TD-MPC2 风格的多层感知机。

    中间层结构为 NormedLinear（Linear + LayerNorm + Mish），
    最后一层根据 act 参数决定：若提供则使用 NormedLinear，
    否则使用裸 nn.Linear。

    参数:
        in_dim: 输入维度。
        mlp_dims: 隐藏层维度列表或单个整数。
        out_dim: 输出维度。
        act: 最后一层的激活函数，为 None 时最后一层
            不带 LayerNorm 和激活。
        dropout: Dropout 概率，仅作用于第一层。

    返回:
        构建好的 nn.Sequential 模型。
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        layers.append(
            NormedLinear(
                dims[i],
                dims[i + 1],
                dropout=dropout * (i == 0),
            )
        )
    # 最后一层
    if act:
        layers.append(
            NormedLinear(dims[-2], dims[-1], act=act)
        )
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)
