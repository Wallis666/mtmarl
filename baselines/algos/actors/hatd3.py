"""HATD3 算法模块。"""

import torch

from baselines.algos.actors.haddpg import HADDPG
from baselines.utils.env import check


class HATD3(HADDPG):
    """HATD3 算法。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 HATD3 算法。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间。
            device: 用于张量运算的设备。
        """
        super().__init__(
            args, obs_space, act_space, device,
        )
        self.policy_noise = args["policy_noise"]
        self.noise_clip = args["noise_clip"]

    def get_target_actions(
        self,
        obs,
    ):
        """
        获取目标 Actor 的动作（带裁剪噪声）。

        参数:
            obs: 目标 Actor 的观测值。

        返回:
            actions: 目标 Actor 采取的动作。
        """
        obs = check(obs).to(**self.tpdv)
        actions = self.target_actor(obs)
        noise = (
            torch.randn_like(actions)
            * self.policy_noise
            * self.scale
        )
        noise = torch.clamp(
            noise,
            -self.noise_clip * self.scale,
            self.noise_clip * self.scale,
        )
        actions += noise
        actions = torch.clamp(
            actions, self.low, self.high,
        )
        return actions
