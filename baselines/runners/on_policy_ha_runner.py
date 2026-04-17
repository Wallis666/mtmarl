"""在策略异构智能体（HA）算法的运行器模块。"""

import numpy as np
import torch

from baselines.runners.on_policy_base_runner import (
    OnPolicyBaseRunner,
)
from baselines.utils.trans import _t2n


class OnPolicyHARunner(OnPolicyBaseRunner):
    """在策略异构智能体算法的运行器。"""

    def train(self):
        """训练模型。"""
        actor_train_infos = []

        # factor 用于考虑前序智能体的更新影响
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # 计算优势函数
        if self.value_normalizer is not None:
            advantages = (
                self.critic_buffer.returns[:-1]
                - self.value_normalizer.denormalize(
                    self.critic_buffer.value_preds[:-1]
                )
            )
        else:
            advantages = (
                self.critic_buffer.returns[:-1]
                - self.critic_buffer.value_preds[:-1]
            )

        # 对 FP 模式归一化优势函数
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks
                for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(
                active_masks_collector,
                axis=2,
            )
            advantages_copy = advantages.copy()
            advantages_copy[
                active_masks_array[:-1] == 0.0
            ] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (
                (advantages - mean_advantages)
                / (std_advantages + 1e-5)
            )

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(
                torch.randperm(self.num_agents).numpy()
            )

        for agent_id in agent_order:
            # 当前演员保存 factor
            self.actor_buffer[agent_id].update_factor(
                factor
            )

            # 以下变形将前两个维度（episode_length 和
            # n_rollout_threads）合并成一个批次
            buf = self.actor_buffer[agent_id]
            available_actions = (
                None
                if buf.available_actions is None
                else buf.available_actions[:-1].reshape(
                    -1,
                    *buf.available_actions.shape[2:],
                )
            )

            # 计算更新前的动作对数概率
            old_actions_logprob, _, _ = (
                self.actor[agent_id].evaluate_actions(
                    buf.obs[:-1].reshape(
                        -1, *buf.obs.shape[2:]
                    ),
                    buf.rnn_states[0:1].reshape(
                        -1, *buf.rnn_states.shape[2:]
                    ),
                    buf.actions.reshape(
                        -1, *buf.actions.shape[2:]
                    ),
                    buf.masks[:-1].reshape(
                        -1, *buf.masks.shape[2:]
                    ),
                    available_actions,
                    buf.active_masks[:-1].reshape(
                        -1, *buf.active_masks.shape[2:]
                    ),
                )
            )

            # 更新演员
            if self.state_type == "EP":
                actor_train_info = (
                    self.actor[agent_id].train(
                        buf,
                        advantages.copy(),
                        "EP",
                    )
                )
            elif self.state_type == "FP":
                actor_train_info = (
                    self.actor[agent_id].train(
                        buf,
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )
                )

            # 计算更新后的动作对数概率
            new_actions_logprob, _, _ = (
                self.actor[agent_id].evaluate_actions(
                    buf.obs[:-1].reshape(
                        -1, *buf.obs.shape[2:]
                    ),
                    buf.rnn_states[0:1].reshape(
                        -1, *buf.rnn_states.shape[2:]
                    ),
                    buf.actions.reshape(
                        -1, *buf.actions.shape[2:]
                    ),
                    buf.masks[:-1].reshape(
                        -1, *buf.masks.shape[2:]
                    ),
                    available_actions,
                    buf.active_masks[:-1].reshape(
                        -1, *buf.active_masks.shape[2:]
                    ),
                )
            )

            # 更新下一个智能体的 factor
            ep_len = self.algo_args["train"][
                "episode_length"
            ]
            n_threads = self.algo_args["train"][
                "n_rollout_threads"
            ]
            agg_fn = getattr(
                torch, self.action_aggregation
            )
            ratio = torch.exp(
                new_actions_logprob - old_actions_logprob
            )
            factor = factor * _t2n(
                agg_fn(ratio, dim=-1).reshape(
                    ep_len, n_threads, 1
                )
            )
            actor_train_infos.append(actor_train_info)

        # 更新评论家
        critic_train_info = self.critic.train(
            self.critic_buffer,
            self.value_normalizer,
        )

        return actor_train_infos, critic_train_info
