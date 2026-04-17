"""价值归一化模块。"""

import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """对观测向量在前 norm_axes 个维度上进行归一化。"""

    def __init__(
        self,
        input_shape,
        norm_axes=1,
        beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"),
    ):
        """
        初始化 ValueNorm。

        参数:
            input_shape: 输入形状。
            norm_axes: 归一化的轴数。
            beta: 指数移动平均系数。
            per_element_update: 是否按元素更新。
            epsilon: 数值稳定性常数。
            device: 用于张量运算的设备。
        """
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape),
            requires_grad=False,
        ).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape),
            requires_grad=False,
        ).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(
            torch.tensor(0.0),
            requires_grad=False,
        ).to(**self.tpdv)

    def running_mean_var(self):
        """获取去偏后的均值和方差。"""
        debiased_mean = (
            self.running_mean
            / self.debiasing_term.clamp(
                min=self.epsilon,
            )
        )
        debiased_mean_sq = (
            self.running_mean_sq
            / self.debiasing_term.clamp(
                min=self.epsilon,
            )
        )
        debiased_var = (
            debiased_mean_sq - debiased_mean**2
        ).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(
        self,
        input_vector,
    ):
        """
        更新运行均值和方差。

        参数:
            input_vector: 输入向量。
        """
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(
                input_vector,
            )
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(
            dim=tuple(range(self.norm_axes)),
        )
        batch_sq_mean = (input_vector**2).mean(
            dim=tuple(range(self.norm_axes)),
        )

        if self.per_element_update:
            batch_size = np.prod(
                input_vector.size()[: self.norm_axes],
            )
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(
            batch_mean * (1.0 - weight),
        )
        self.running_mean_sq.mul_(weight).add_(
            batch_sq_mean * (1.0 - weight),
        )
        self.debiasing_term.mul_(weight).add_(
            1.0 * (1.0 - weight),
        )

    def normalize(
        self,
        input_vector,
    ):
        """
        对输入向量进行归一化。

        参数:
            input_vector: 输入向量。

        返回:
            归一化后的向量。
        """
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(
                input_vector,
            )
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
            input_vector
            - mean[(None,) * self.norm_axes]
        ) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(
        self,
        input_vector,
    ):
        """
        将归一化数据还原为原始分布。

        参数:
            input_vector: 归一化后的向量。

        返回:
            还原后的数组。
        """
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(
                input_vector,
            )
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
            input_vector
            * torch.sqrt(var)[
                (None,) * self.norm_axes
            ]
            + mean[(None,) * self.norm_axes]
        )

        out = out.cpu().numpy()

        return out
