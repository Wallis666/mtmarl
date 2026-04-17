"""HADDPG 算法模块。"""

from copy import deepcopy

import torch

from baselines.algos.actors.off_policy_base import OffPolicyBase
from baselines.models.policy.deterministic_policy import (
    DeterministicPolicy,
)
from baselines.utils.env import check


class HADDPG(OffPolicyBase):
    """HADDPG 算法。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 HADDPG 算法。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间（仅支持连续动作空间）。
            device: 用于张量运算的设备。
        """
        assert (
            act_space.__class__.__name__ == "Box"
        ), (
            f"{self.__class__.__name__}"
            " 仅支持连续动作空间。"
        )
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.expl_noise = args["expl_noise"]

        self.actor = DeterministicPolicy(
            args, obs_space, act_space, device,
        )
        self.target_actor = deepcopy(self.actor)
        for p in self.target_actor.parameters():
            p.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr,
        )
        self.low = torch.tensor(
            act_space.low,
        ).to(**self.tpdv)
        self.high = torch.tensor(
            act_space.high,
        ).to(**self.tpdv)
        self.scale = (self.high - self.low) / 2
        self.mean = (self.high + self.low) / 2
        self.turn_off_grad()

    def get_actions(
        self,
        obs,
        add_noise,
    ):
        """
        根据观测获取动作。

        参数:
            obs: Actor 的观测值。
            add_noise: 是否添加探索噪声。

        返回:
            actions: Actor 采取的动作。
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.actor(obs)
        if add_noise:
            actions += (
                torch.randn_like(actions)
                * self.expl_noise
                * self.scale
            )
            actions = torch.clamp(
                actions, self.low, self.high,
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
        return self.target_actor(obs)
