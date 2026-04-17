"""离策略多智能体（MA）算法的运行器模块。"""

import copy

import torch

from baselines.runners.off_policy_base_runner import (
    OffPolicyBaseRunner,
)


class OffPolicyMARunner(OffPolicyBaseRunner):
    """离策略多智能体算法的运行器。"""

    def train(self):
        """训练模型。"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            # EP: (batch_size, dim)
            # FP: (n_agents * batch_size, dim)
            sp_share_obs,
            # (n_agents, batch_size, dim)
            sp_obs,
            # (n_agents, batch_size, dim)
            sp_actions,
            # (n_agents, batch_size, dim)
            sp_available_actions,
            # EP: (batch_size, 1)
            # FP: (n_agents * batch_size, 1)
            sp_reward,
            # EP: (batch_size, 1)
            # FP: (n_agents * batch_size, 1)
            sp_done,
            # (n_agents, batch_size, 1)
            sp_valid_transition,
            # EP: (batch_size, 1)
            # FP: (n_agents * batch_size, 1)
            sp_term,
            # EP: (batch_size, dim)
            # FP: (n_agents * batch_size, dim)
            sp_next_share_obs,
            # (n_agents, batch_size, dim)
            sp_next_obs,
            # (n_agents, batch_size, dim)
            sp_next_available_actions,
            # EP: (batch_size, 1)
            # FP: (n_agents * batch_size, 1)
            sp_gamma,
        ) = data

        # 训练评论家
        self.critic.turn_on_grad()
        next_actions = []
        for agent_id in range(self.num_agents):
            next_actions.append(
                self.actor[
                    agent_id
                ].get_target_actions(
                    sp_next_obs[agent_id]
                )
            )
        self.critic.train(
            sp_share_obs,
            sp_actions,
            sp_reward,
            sp_done,
            sp_term,
            sp_next_share_obs,
            next_actions,
            sp_gamma,
        )
        self.critic.turn_off_grad()

        if self.total_it % self.policy_freq == 0:
            # 训练演员
            # actions 形状: (n_agents, batch_size, dim)
            for agent_id in range(self.num_agents):
                actions = copy.deepcopy(
                    torch.tensor(sp_actions)
                ).to(self.device)
                self.actor[agent_id].turn_on_grad()
                # 训练该智能体
                actions[agent_id] = self.actor[
                    agent_id
                ].get_actions(
                    sp_obs[agent_id], False
                )
                actions_list = [a for a in actions]
                actions_t = torch.cat(
                    actions_list, dim=-1
                )
                value_pred = self.critic.get_values(
                    sp_share_obs, actions_t
                )
                actor_loss = -torch.mean(value_pred)
                self.actor[
                    agent_id
                ].actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor[
                    agent_id
                ].actor_optimizer.step()
                self.actor[agent_id].turn_off_grad()

            # 软更新
            for agent_id in range(self.num_agents):
                self.actor[agent_id].soft_update()
            self.critic.soft_update()
