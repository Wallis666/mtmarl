"""V 网络模块。"""

import torch
import torch.nn as nn

from baselines.models.base.cnn import CNNBase
from baselines.models.base.mlp import MLPBase
from baselines.models.base.rnn import RNNLayer
from baselines.utils.env import (
    check,
    get_shape_from_obs_space,
)
from baselines.utils.model import init, get_init_method


class VNet(nn.Module):
    """V 网络，根据全局状态输出价值函数预测。"""

    def __init__(
        self,
        args,
        cent_obs_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 VNet 模型。

        参数:
            args: 包含模型相关信息的参数字典。
            cent_obs_space: 中心化观测空间。
            device: 运行设备。
        """
        super(VNet, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = (
            args["initialization_method"]
        )
        self.use_naive_recurrent_policy = (
            args["use_naive_recurrent_policy"]
        )
        self.use_recurrent_policy = (
            args["use_recurrent_policy"]
        )
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        init_method = get_init_method(
            self.initialization_method,
        )

        cent_obs_shape = get_shape_from_obs_space(
            cent_obs_space,
        )
        base = (
            CNNBase
            if len(cent_obs_shape) == 3
            else MLPBase
        )
        self.base = base(args, cent_obs_shape)

        if (
            self.use_naive_recurrent_policy
            or self.use_recurrent_policy
        ):
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        def init_(m):
            return init(
                m,
                init_method,
                lambda x: nn.init.constant_(x, 0),
            )

        self.v_out = init_(
            nn.Linear(self.hidden_sizes[-1], 1),
        )

        self.to(device)

    def forward(
        self,
        cent_obs,
        rnn_states,
        masks,
    ):
        """
        根据给定输入计算价值。

        参数:
            cent_obs: 中心化观测输入。
            rnn_states: RNN 隐藏状态。
            masks: 指示 RNN 状态是否需要重置的掩码。

        返回:
            values: 价值函数预测值。
            rnn_states: 更新后的 RNN 隐藏状态。
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(
            **self.tpdv,
        )
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if (
            self.use_naive_recurrent_policy
            or self.use_recurrent_policy
        ):
            critic_features, rnn_states = self.rnn(
                critic_features, rnn_states, masks,
            )
        values = self.v_out(critic_features)

        return values, rnn_states
