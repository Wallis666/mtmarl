"""On-Policy 算法基类模块。"""

import torch

from baselines.models.policy.stochastic_policy import StochasticPolicy
from baselines.utils.model import update_linear_schedule


class OnPolicyBase:
    """On-Policy 算法基类。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 On-Policy 算法基类。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间。
            device: 用于张量运算的设备。
        """
        # 保存参数
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = (
            args["use_recurrent_policy"]
        )
        self.use_naive_recurrent_policy = (
            args["use_naive_recurrent_policy"]
        )
        self.use_policy_active_masks = (
            args["use_policy_active_masks"]
        )
        self.action_aggregation = (
            args["action_aggregation"]
        )

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        # 保存观测空间和动作空间
        self.obs_space = obs_space
        self.act_space = act_space
        # 创建 Actor 网络
        self.actor = StochasticPolicy(
            args, self.obs_space, self.act_space, self.device,
        )
        # 创建 Actor 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(
        self,
        episode,
        episodes,
    ):
        """
        衰减学习率。

        参数:
            episode: 当前训练回合数。
            episodes: 总训练回合数。
        """
        update_linear_schedule(
            self.actor_optimizer,
            episode,
            episodes,
            self.lr,
        )

    def get_actions(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        """
        根据给定输入计算动作。

        参数:
            obs: Actor 的局部观测输入。
            rnn_states_actor: Actor 的 RNN 状态。
            masks: 指示 RNN 状态是否需要重置的掩码。
            available_actions: 可用动作掩码。
            deterministic: 是否使用确定性动作。
        """
        actions, action_log_probs, rnn_states_actor = (
            self.actor(
                obs,
                rnn_states_actor,
                masks,
                available_actions,
                deterministic,
            )
        )
        return actions, action_log_probs, rnn_states_actor

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """
        获取动作的对数概率、熵和分布，用于 Actor 更新。

        参数:
            obs: Actor 的局部观测输入。
            rnn_states_actor: Actor 的 RNN 状态。
            action: 需要计算对数概率和熵的动作。
            masks: 指示 RNN 状态是否需要重置的掩码。
            available_actions: 可用动作掩码。
            active_masks: 指示智能体是否存活的掩码。
        """
        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.actor.evaluate_actions(
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions,
            active_masks,
        )
        return (
            action_log_probs,
            dist_entropy,
            action_distribution,
        )

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        """
        根据给定输入计算动作。

        参数:
            obs: Actor 的局部观测输入。
            rnn_states_actor: Actor 的 RNN 状态。
            masks: 指示 RNN 状态是否需要重置的掩码。
            available_actions: 可用动作掩码。
            deterministic: 是否使用确定性动作。
        """
        actions, _, rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states_actor

    def update(
        self,
        sample,
    ):
        """
        更新 Actor 网络。

        参数:
            sample: 包含用于更新网络的数据批次。
        """
        pass

    def train(
        self,
        actor_buffer,
        advantages,
        state_type,
    ):
        """
        使用小批量梯度下降执行一次训练更新。

        参数:
            actor_buffer: 包含 Actor 训练数据的缓冲区。
            advantages: 优势值数组。
            state_type: 状态类型。
        """
        pass

    def prep_training(self):
        """切换到训练模式。"""
        self.actor.train()

    def prep_rollout(self):
        """切换到评估模式。"""
        self.actor.eval()
