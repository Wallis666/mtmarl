"""压缩高斯策略网络模块，用于 HASAC 算法。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from baselines.models.base.plain_cnn import PlainCNN
from baselines.models.base.plain_mlp import PlainMLP
from baselines.utils.env import get_shape_from_obs_space

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianPolicy(nn.Module):
    """用于 HASAC 的压缩高斯策略网络。"""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        """初始化压缩高斯策略模型。

        Args:
            args: 包含模型相关信息的字典。
            obs_space: 观测空间。
            action_space: 动作空间。
            device: 指定运行设备（cpu/gpu）。
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3:
            self.feature_extractor = PlainCNN(
                obs_shape,
                hidden_sizes[0],
                activation_func,
            )
            feature_dim = hidden_sizes[0]
        else:
            self.feature_extractor = None
            feature_dim = obs_shape[0]
        act_dim = action_space.shape[0]
        self.net = PlainMLP(
            [feature_dim] + list(hidden_sizes),
            activation_func,
            final_activation_func,
        )
        self.mu_layer = nn.Linear(
            hidden_sizes[-1],
            act_dim,
        )
        self.log_std_layer = nn.Linear(
            hidden_sizes[-1],
            act_dim,
        )
        # 动作限幅值（假设所有维度共享相同的边界）
        self.act_limit = action_space.high[0]
        self.to(device)

    def forward(
        self,
        obs,
        stochastic=True,
        with_logprob=True,
    ):
        """前向传播，返回缩放到动作空间范围内的输出。"""
        if self.feature_extractor is not None:
            x = self.feature_extractor(obs)
        else:
            x = obs
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(
            log_std,
            LOG_STD_MIN,
            LOG_STD_MAX,
        )
        std = torch.exp(log_std)

        # 压缩前的分布与采样
        pi_distribution = Normal(mu, std)
        if not stochastic:
            # 仅在测试时评估策略使用
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # 从高斯分布计算对数概率，然后应用 Tanh 压缩
            # 的修正项。
            # 注意：修正公式较为巧妙。要理解其来源，请参阅
            # 原始 SAC 论文（arXiv 1801.01290）附录 C。
            # 此为等式 21 的数值更稳定的等价形式。
            logp_pi = pi_distribution.log_prob(
                pi_action
            ).sum(axis=-1, keepdim=True)
            logp_pi -= (
                2
                * (
                    np.log(2)
                    - pi_action
                    - F.softplus(-2 * pi_action)
                )
            ).sum(axis=1, keepdim=True)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
