"""
动力学模型模块。

提供基于 Soft MoE 的集中式动力学模型，用于在 latent
空间中预测多智能体的下一步状态。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.mlp import mlp


class SoftMoEDynamics(nn.Module):
    """
    基于 Soft MoE 的集中式动力学模型。

    将所有 agent 的 latent 状态和动作拼接后，通过
    Soft MoE 路由机制分配给多个 expert 并行处理，
    最终加权合并得到每个 agent 的下一步 latent 状态。

    Soft MoE 的核心流程：
        1. Dispatch：在 agent 维度上 softmax，将
           agent token 加权混合后分配给各 expert。
        2. Expert Forward：每个 expert 独立处理
           分配到的输入。
        3. Combine：在 expert 维度上 softmax，将
           各 expert 的输出加权合并回每个 agent。
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        mlp_dims: list[int],
        num_experts: int,
        act: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        """
        初始化 SoftMoEDynamics。

        参数:
            latent_dim: latent 状态维度。
            action_dim: 单个 agent 的动作维度。
            mlp_dims: expert MLP 的隐藏层维度列表。
            num_experts: expert 数量。
            act: expert MLP 最后一层的激活函数。
                应与 encoder 的输出激活保持一致
                （如 SimNorm），以确保预测的 latent
                与编码的 latent 处于同一空间。
            dropout: Dropout 概率，仅作用于 expert
                MLP 的第一层。
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.d_model = latent_dim + action_dim
        self.num_experts = num_experts

        # 每个 expert 是一个独立的 MLP
        # 输入: (latent_dim + action_dim) → 输出: latent_dim
        self.experts = nn.ModuleList([
            mlp(
                in_dim=self.d_model,
                mlp_dims=mlp_dims,
                out_dim=latent_dim,
                act=act,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

        # 路由参数 phi: (d_model, num_experts, num_slots)
        # num_slots=1，每个 expert 一个 slot
        self.phi = nn.Parameter(
            torch.randn(self.d_model, num_experts, 1)
            * (1 / math.sqrt(self.d_model))
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测下一步 latent 状态。

        参数:
            z: 当前 latent 状态，
                形状为 (B, N_agents, latent_dim)。
            a: 当前动作，
                形状为 (B, N_agents, action_dim)。

        返回:
            预测的下一步 latent 状态，
            形状为 (B, N_agents, latent_dim)。
        """
        # 拼接状态和动作
        # x: (B, N_agents, d_model)
        x = torch.cat([z, a], dim=-1)

        # 计算路由权重
        # weights: (B, N_agents, num_experts, num_slots)
        weights = torch.einsum(
            "b n d, d e s -> b n e s", x, self.phi,
        )

        # --- Dispatch: agent → expert ---
        # 在 agent 维度 (dim=1) 上做 softmax，
        # 决定每个 expert 从各 agent 取多少信息
        dispatch_w = F.softmax(weights, dim=1)
        # expert 输入 = dispatch 权重加权的 agent token
        # expert_inputs: (B, num_experts, num_slots, d_model)
        expert_inputs = torch.einsum(
            "b n e s, b n d -> b e s d", dispatch_w, x,
        )

        # --- Expert Forward ---
        # 每个 expert 独立处理各自的输入
        # expert_outputs: (num_experts, B, num_slots, latent_dim)
        expert_outputs = torch.stack([
            self.experts[i](expert_inputs[:, i])
            for i in range(self.num_experts)
        ])
        # 重排为 (B, num_experts * num_slots, latent_dim)
        expert_outputs = expert_outputs.permute(
            1, 0, 2, 3,
        ).reshape(x.shape[0], -1, self.latent_dim)

        # --- Combine: expert → agent ---
        # 在 expert 维度 (dim=-1) 上做 softmax，
        # 决定每个 agent 从各 expert 取多少结果
        combine_w = weights.reshape(
            x.shape[0], x.shape[1], -1,
        )
        combine_w = F.softmax(combine_w, dim=-1)
        # 加权合并 expert 输出回每个 agent
        # out: (B, N_agents, latent_dim)
        out = torch.einsum(
            "b n e, b e d -> b n d",
            combine_w, expert_outputs,
        )

        return out
