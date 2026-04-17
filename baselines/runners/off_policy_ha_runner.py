"""离策略异构智能体（HA）算法的运行器模块。"""

import numpy as np
import torch
import torch.nn.functional as F

from baselines.runners.off_policy_base_runner import (
    OffPolicyBaseRunner,
)


class OffPolicyHARunner(OffPolicyBaseRunner):
    """离策略异构智能体算法的运行器。"""

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
        if self.args["algo"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                avail = (
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions
                    is not None
                    else None
                )
                next_action, next_logp_action = (
                    self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_next_obs[agent_id],
                        avail,
                    )
                )
                next_actions.append(next_action)
                next_logp_actions.append(
                    next_logp_action
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
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

        sp_valid_transition = torch.tensor(
            sp_valid_transition,
            device=self.device,
        )

        if self.total_it % self.policy_freq == 0:
            # 训练演员
            if self.args["algo"] == "hasac":
                self._train_hasac(
                    sp_obs,
                    sp_share_obs,
                    sp_available_actions,
                    sp_valid_transition,
                    next_logp_actions,
                )
            else:
                if self.args["algo"] == "had3qn":
                    self._train_had3qn(
                        sp_obs,
                        sp_share_obs,
                    )
                else:
                    self._train_haddpg(
                        sp_obs,
                        sp_share_obs,
                    )
                # 软更新
                for agent_id in range(
                    self.num_agents
                ):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()

    def _train_hasac(
        self,
        sp_obs,
        sp_share_obs,
        sp_available_actions,
        sp_valid_transition,
        next_logp_actions,
    ):
        """HASAC 算法的演员训练逻辑。"""
        actions = []
        logp_actions = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                avail = (
                    sp_available_actions[agent_id]
                    if sp_available_actions is not None
                    else None
                )
                action, logp_action = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_obs[agent_id],
                    avail,
                )
                actions.append(action)
                logp_actions.append(logp_action)

        # actions 形状: (n_agents, batch_size, dim)
        # logp_actions 形状: (n_agents, batch_size, 1)
        if self.fixed_order:
            agent_order = list(
                range(self.num_agents)
            )
        else:
            agent_order = list(
                np.random.permutation(self.num_agents)
            )

        for agent_id in agent_order:
            self.actor[agent_id].turn_on_grad()
            avail = (
                sp_available_actions[agent_id]
                if sp_available_actions is not None
                else None
            )
            # 训练该智能体
            (
                actions[agent_id],
                logp_actions[agent_id],
            ) = self.actor[
                agent_id
            ].get_actions_with_logprobs(
                sp_obs[agent_id],
                avail,
            )

            if self.state_type == "EP":
                logp_action = logp_actions[agent_id]
                actions_t = torch.cat(
                    actions, dim=-1
                )
            elif self.state_type == "FP":
                logp_action = torch.tile(
                    logp_actions[agent_id],
                    (self.num_agents, 1),
                )
                actions_t = torch.tile(
                    torch.cat(actions, dim=-1),
                    (self.num_agents, 1),
                )

            value_pred = self.critic.get_values(
                sp_share_obs, actions_t
            )

            use_masks = self.algo_args["algo"][
                "use_policy_active_masks"
            ]
            if use_masks:
                if self.state_type == "EP":
                    vt = sp_valid_transition[
                        agent_id
                    ]
                    actor_loss = (
                        -torch.sum(
                            (
                                value_pred
                                - self.alpha[agent_id]
                                * logp_action
                            )
                            * vt
                        )
                        / vt.sum()
                    )
                elif self.state_type == "FP":
                    vt = torch.tile(
                        sp_valid_transition[
                            agent_id
                        ],
                        (self.num_agents, 1),
                    )
                    actor_loss = (
                        -torch.sum(
                            (
                                value_pred
                                - self.alpha[agent_id]
                                * logp_action
                            )
                            * vt
                        )
                        / vt.sum()
                    )
            else:
                actor_loss = -torch.mean(
                    value_pred
                    - self.alpha[agent_id]
                    * logp_action
                )

            self.actor[
                agent_id
            ].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[
                agent_id
            ].actor_optimizer.step()
            self.actor[agent_id].turn_off_grad()

            # 训练该智能体的 alpha
            if self.algo_args["algo"]["auto_alpha"]:
                log_prob = (
                    logp_actions[agent_id].detach()
                    + self.target_entropy[agent_id]
                )
                alpha_loss = -(
                    self.log_alpha[agent_id]
                    * log_prob
                ).mean()
                self.alpha_optimizer[
                    agent_id
                ].zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer[
                    agent_id
                ].step()
                self.alpha[agent_id] = torch.exp(
                    self.log_alpha[
                        agent_id
                    ].detach()
                )

            actions[agent_id], _ = self.actor[
                agent_id
            ].get_actions_with_logprobs(
                sp_obs[agent_id],
                avail,
            )

        # 训练评论家的 alpha
        if self.algo_args["algo"]["auto_alpha"]:
            self.critic.update_alpha(
                logp_actions,
                np.sum(self.target_entropy),
            )

    def _train_had3qn(
        self,
        sp_obs,
        sp_share_obs,
    ):
        """HAD3QN 算法的演员训练逻辑。"""
        actions = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                actions.append(
                    self.actor[
                        agent_id
                    ].get_actions(
                        sp_obs[agent_id], False
                    )
                )

        # actions 形状: (n_agents, batch_size, 1)
        update_actions, get_values = (
            self.critic.train_values(
                sp_share_obs, actions
            )
        )

        if self.fixed_order:
            agent_order = list(
                range(self.num_agents)
            )
        else:
            agent_order = list(
                np.random.permutation(self.num_agents)
            )

        for agent_id in agent_order:
            self.actor[agent_id].turn_on_grad()
            # 演员预测值
            actor_values = self.actor[
                agent_id
            ].train_values(
                sp_obs[agent_id],
                actions[agent_id],
            )
            # 评论家预测值
            critic_values = get_values()
            # 更新
            actor_loss = torch.mean(
                F.mse_loss(actor_values, critic_values)
            )
            self.actor[
                agent_id
            ].actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor[
                agent_id
            ].actor_optimizer.step()
            self.actor[agent_id].turn_off_grad()
            update_actions(agent_id)

    def _train_haddpg(
        self,
        sp_obs,
        sp_share_obs,
    ):
        """HADDPG 算法的演员训练逻辑。"""
        actions = []
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                actions.append(
                    self.actor[
                        agent_id
                    ].get_actions(
                        sp_obs[agent_id], False
                    )
                )

        # actions 形状: (n_agents, batch_size, dim)
        if self.fixed_order:
            agent_order = list(
                range(self.num_agents)
            )
        else:
            agent_order = list(
                np.random.permutation(self.num_agents)
            )

        for agent_id in agent_order:
            self.actor[agent_id].turn_on_grad()
            # 训练该智能体
            actions[agent_id] = self.actor[
                agent_id
            ].get_actions(
                sp_obs[agent_id], False
            )
            actions_t = torch.cat(
                actions, dim=-1
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
            actions[agent_id] = self.actor[
                agent_id
            ].get_actions(
                sp_obs[agent_id], False
            )
