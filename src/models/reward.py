"""
奖励模型模块。

提供基于 Sparse MoE 的集中式奖励模型，用于在 latent
空间中预测多智能体的全局奖励。每个 expert 使用
Self-Attention 处理 agent 间交互，由 Noisy Top-K
Router 稀疏选择被激活的 expert。
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class SelfAttnExpert(nn.Module):
    """
    基于 Self-Attention 的 expert 模块。

    结构为标准 Transformer block：
    Self-Attention + 残差 + LayerNorm + FFN + 残差 + LayerNorm。
    用于捕获 agent 之间的交互关系。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ffn_hidden: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        """
        初始化 SelfAttnExpert。

        参数:
            d_model: 输入和输出的特征维度。
            num_heads: 注意力头数。
            ffn_hidden: FFN 中间层维度。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入张量，形状为 (B, N_agents, d_model)。

        返回:
            输出张量，形状为 (B, N_agents, d_model)。
        """
        attn_out, _ = self.attn(
            x, x, x, need_weights=False,
        )
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class NoisyTopKRouter(nn.Module):
    """
    带噪声的 Top-K 路由器。

    为每个输入计算各 expert 的门控权重，仅保留
    top-k 个 expert 的权重，其余置零。训练时在
    logits 上添加可学习的噪声以促进探索。

    附带负载均衡损失，防止所有输入集中到少数
    expert 上：
        balance_loss = CV²(importance) + CV²(load)
                       + z_loss(logits)
    """

    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        top_k: int = 2,
        noisy_gating: bool = True,
    ) -> None:
        """
        初始化 NoisyTopKRouter。

        参数:
            in_dim: 输入特征维度。
            num_experts: expert 总数。
            top_k: 每个输入激活的 expert 数量。
            noisy_gating: 训练时是否添加门控噪声。
        """
        super().__init__()
        assert top_k <= num_experts, (
            f"top_k ({top_k}) 不能超过 "
            f"num_experts ({num_experts})"
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(
            torch.zeros(in_dim, num_experts),
        )
        self.w_noise = nn.Parameter(
            torch.zeros(in_dim, num_experts),
        )
        self.register_buffer(
            "mean", torch.tensor([0.0]),
        )
        self.register_buffer(
            "std", torch.tensor([1.0]),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        计算门控权重和负载均衡损失。

        参数:
            x: 输入特征，形状为 (B, in_dim)。

        返回:
            gates: 门控权重，形状为 (B, num_experts)，
                仅 top-k 个非零。
            aux: 辅助信息字典，包含:
                - balance_loss: 负载均衡损失标量。
                - importance: 各 expert 的重要度。
                - load: 各 expert 的负载。
        """
        # 计算 logits
        clean_logits = x @ self.w_gate  # (B, N_e)
        if self.noisy_gating and self.training:
            noise_std = (
                nn.functional.softplus(x @ self.w_noise)
                + 1e-2
            )
            noisy_logits = (
                clean_logits
                + torch.randn_like(clean_logits) * noise_std
            )
            logits = noisy_logits
        else:
            noise_std = None
            noisy_logits = None
            logits = clean_logits

        # 选出 top-k（多取一个用于负载估计）
        num_topk = min(self.top_k + 1, self.num_experts)
        top_logits, top_indices = logits.topk(
            num_topk, dim=1,
        )
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = torch.softmax(top_k_logits, dim=1)

        # 构造稀疏门控向量
        gates = torch.zeros_like(logits, requires_grad=True)
        gates = gates.scatter(
            1, top_k_indices, top_k_gates,
        )

        # 计算负载均衡损失
        if (self.noisy_gating
                and self.top_k < self.num_experts
                and self.training):
            load = self._prob_in_top_k(
                clean_logits, noisy_logits,
                noise_std, top_logits,
            ).sum(0)
        else:
            load = (gates > 0).float().sum(0)

        importance = gates.sum(0)
        balance_loss = (
            self._cv_squared(importance)
            + self._cv_squared(load)
            + self._z_loss(logits)
        )

        aux = dict(
            balance_loss=balance_loss,
            importance=importance,
            load=load,
        )
        return gates, aux

    def _prob_in_top_k(
        self,
        clean_logits: torch.Tensor,
        noisy_logits: torch.Tensor,
        noise_std: torch.Tensor,
        top_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        估计各 expert 被选入 top-k 的概率。

        参数:
            clean_logits: 无噪声 logits，(B, N_e)。
            noisy_logits: 含噪声 logits，(B, N_e)。
            noise_std: 噪声标准差，(B, N_e)。
            top_logits: top-(k+1) logits，
                (B, k+1)。

        返回:
            各 expert 的入选概率，(B, N_e)。
        """
        batch = clean_logits.size(0)
        m = top_logits.size(1)
        top_flat = top_logits.flatten()

        # 若在 top-k 内，阈值是第 k+1 位
        pos_in = (
            torch.arange(batch, device=clean_logits.device)
            * m + self.top_k
        )
        threshold_in = top_flat.gather(
            0, pos_in,
        ).unsqueeze(1)
        is_in = noisy_logits.gt(threshold_in)

        # 若在 top-k 外，阈值是第 k 位
        pos_out = pos_in - 1
        threshold_out = top_flat.gather(
            0, pos_out,
        ).unsqueeze(1)

        normal = Normal(self.mean, self.std)
        prob_in = normal.cdf(
            (clean_logits - threshold_in) / noise_std,
        )
        prob_out = normal.cdf(
            (clean_logits - threshold_out) / noise_std,
        )
        return torch.where(is_in, prob_in, prob_out)

    @staticmethod
    def _cv_squared(x: torch.Tensor) -> torch.Tensor:
        """计算变异系数的平方 CV²。"""
        if x.shape[0] == 1:
            return torch.tensor(
                0.0, device=x.device, dtype=x.dtype,
            )
        mean = x.float().mean()
        return x.float().var() / (mean ** 2 + 1e-10)

    @staticmethod
    def _z_loss(logits: torch.Tensor) -> torch.Tensor:
        """计算 z-loss，惩罚过大的 logits。"""
        return torch.logsumexp(logits, dim=-1).mean()


class SparseMoEReward(nn.Module):
    """
    基于 Sparse MoE 的集中式奖励模型。

    核心流程：
        1. 拼接 (z, a) 并展平为路由输入。
        2. NoisyTopKRouter 选择 top-k 个 expert。
        3. 仅被选中的 SelfAttnExpert 处理输入，
           按门控权重加权聚合。
        4. Reward head 映射到奖励 logits。
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_agents: int,
        num_experts: int,
        top_k: int = 2,
        num_bins: int = 101,
        num_heads: int = 1,
        ffn_hidden: int = 1024,
        head_hidden: int = 512,
        expert_dropout: float = 0.0,
        noisy_gating: bool = True,
    ) -> None:
        """
        初始化 SparseMoEReward。

        参数:
            latent_dim: latent 状态维度。
            action_dim: 单个 agent 的动作维度。
            num_agents: agent 数量。
            num_experts: expert 总数。
            top_k: 每个输入激活的 expert 数量。
            num_bins: 奖励输出维度。使用 TwoHot
                编码时为 bin 数量，否则为 1。
            num_heads: Self-Attention 头数。
            ffn_hidden: expert FFN 中间层维度。
            head_hidden: reward head 中间层维度。
            expert_dropout: expert 内部 Dropout 概率。
            noisy_gating: 训练时是否添加门控噪声。
        """
        super().__init__()
        self.d_model = latent_dim + action_dim
        self.num_agents = num_agents
        self.num_experts = num_experts

        self.router = NoisyTopKRouter(
            in_dim=num_agents * self.d_model,
            num_experts=num_experts,
            top_k=top_k,
            noisy_gating=noisy_gating,
        )

        self.experts = nn.ModuleList([
            SelfAttnExpert(
                d_model=self.d_model,
                num_heads=num_heads,
                ffn_hidden=ffn_hidden,
                dropout=expert_dropout,
            )
            for _ in range(num_experts)
        ])

        self.reward_head = nn.Sequential(
            nn.Linear(num_agents * self.d_model, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_bins),
        )

    def forward(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        预测奖励 logits。

        参数:
            z: 当前 latent 状态，
                形状为 (B, N_agents, latent_dim)。
            a: 当前动作，
                形状为 (B, N_agents, action_dim)。

        返回:
            r_logits: 奖励 logits，
                形状为 (B, num_bins)。
            aux: 辅助信息字典，包含:
                - balance_loss: 负载均衡损失。
                - gates: 门控权重。
        """
        b = z.shape[0]
        # 拼接状态和动作
        # x: (B, N_agents, d_model)
        x = torch.cat([z, a], dim=-1)

        # --- Router: 选择 top-k expert ---
        # 展平所有 agent 信息作为路由输入
        x_flat = x.reshape(b, -1)  # (B, N_agents * d_model)
        gates, aux = self.router(x_flat)  # (B, N_e)

        # --- Sparse Expert Forward ---
        # 只计算被选中的 expert，按门控权重加权
        y = torch.zeros_like(x)  # (B, N_agents, d_model)
        for e_idx, expert in enumerate(self.experts):
            mask = gates[:, e_idx] > 0
            if not mask.any():
                continue
            out_e = expert(x[mask])  # (b_i, N_agents, d_model)
            w = gates[mask, e_idx].view(-1, 1, 1)
            y[mask] = y[mask] + w * out_e

        # --- Reward Head ---
        y_flat = y.reshape(b, -1)  # (B, N_agents * d_model)
        r_logits = self.reward_head(y_flat)  # (B, num_bins)

        aux["gates"] = gates
        return r_logits, aux
