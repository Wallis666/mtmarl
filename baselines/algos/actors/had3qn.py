"""HAD3QN 算法模块。"""

from copy import deepcopy

import numpy as np
import torch

from baselines.algos.actors.off_policy_base import OffPolicyBase
from baselines.models.value.dueling_q_net import DuelingQNet
from baselines.utils.env import check


class HAD3QN(OffPolicyBase):
    """HAD3QN 算法。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 HAD3QN 算法。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间（仅支持离散动作空间）。
            device: 用于张量运算的设备。
        """
        assert (
            act_space.__class__.__name__ == "Discrete"
        ), "HAD3QN 仅支持离散动作空间。"
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        self.tpdv_a = dict(
            dtype=torch.int64, device=device,
        )
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.epsilon = args["epsilon"]
        self.action_dim = act_space.n

        self.actor = DuelingQNet(
            args, obs_space, self.action_dim, device,
        )
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr,
        )
        self.turn_off_grad()

    def get_actions(
        self,
        obs,
        epsilon_greedy,
    ):
        """
        根据观测获取动作。

        参数:
            obs: Actor 的观测值。
            epsilon_greedy: 是否使用 epsilon 贪心策略。

        返回:
            actions: Actor 采取的动作。
        """
        obs = check(obs).to(**self.tpdv)
        if (
            np.random.random() < self.epsilon
            and epsilon_greedy
        ):
            actions = torch.randint(
                low=0,
                high=self.action_dim,
                size=(*obs.shape[:-1], 1),
            )
        else:
            actions = self.actor(obs).argmax(
                dim=-1, keepdim=True,
            )
        return actions

    def get_target_actions(
        self,
        obs,
    ):
        """
        获取目标 Actor 的动作。

        参数:
            obs: 目标 Actor 的观测值。

        返回:
            actions: 目标 Actor 采取的动作。
        """
        obs = check(obs).to(**self.tpdv)
        return self.target_actor(obs).argmax(
            dim=-1, keepdim=True,
        )

    def train_values(
        self,
        obs,
        actions,
    ):
        """
        获取带梯度的 Q 值。

        参数:
            obs: 观测值批次。
            actions: 动作批次。

        返回:
            values: Q 网络预测的价值。
        """
        obs = check(obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv_a)
        values = torch.gather(
            input=self.actor(obs),
            dim=1,
            index=actions,
        )
        return values
