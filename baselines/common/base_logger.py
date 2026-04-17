"""基础日志记录器模块。"""

import os
import time

import numpy as np


class BaseLogger:
    """基础日志记录器，用于 On-Policy 训练流程的信息记录。"""

    def __init__(
        self,
        args,
        algo_args,
        env_args,
        num_agents,
        writter,
        run_dir,
    ):
        """
        初始化日志记录器。

        参数:
            args: 全局参数字典。
            algo_args: 算法参数字典。
            env_args: 环境参数字典。
            num_agents: 智能体数量。
            writter: TensorBoard 写入器。
            run_dir: 运行目录路径。
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"),
            "w",
            encoding="utf-8",
        )

    def get_task_name(self):
        """获取任务名称。"""
        raise NotImplementedError

    def init(
        self,
        episodes,
    ):
        """
        初始化日志记录器。

        参数:
            episodes: 总训练回合数。
        """
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"],
        )
        self.done_episodes_rewards = []

    def episode_init(
        self,
        episode,
    ):
        """
        每个回合开始时初始化日志记录器。

        参数:
            episode: 当前回合数。
        """
        self.episode = episode

    def per_step(
        self,
        data,
    ):
        """
        每步处理数据。

        参数:
            data: 包含观测、奖励等信息的元组。
        """
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        dones_env = np.all(dones, axis=1)
        reward_env = np.mean(rewards, axis=1).flatten()
        self.train_episode_rewards += reward_env
        n_threads = (
            self.algo_args["train"]["n_rollout_threads"]
        )
        for t in range(n_threads):
            if dones_env[t]:
                self.done_episodes_rewards.append(
                    self.train_episode_rewards[t],
                )
                self.train_episode_rewards[t] = 0

    def episode_log(
        self,
        actor_train_infos,
        critic_train_info,
        actor_buffer,
        critic_buffer,
    ):
        """
        每个回合结束时记录日志。

        参数:
            actor_train_infos: Actor 训练信息列表。
            critic_train_info: Critic 训练信息字典。
            actor_buffer: Actor 缓冲区。
            critic_buffer: Critic 缓冲区。
        """
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"][
                "n_rollout_threads"
            ]
        )
        self.end = time.time()
        print(
            "环境 {} 任务 {} 算法 {} 实验 {} "
            "更新 {}/{} 回合, "
            "总时间步 {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"][
                    "num_env_steps"
                ],
                int(
                    self.total_num_steps
                    / (self.end - self.start)
                ),
            )
        )

        critic_train_info["average_step_rewards"] = (
            critic_buffer.get_mean_rewards()
        )
        self.log_train(
            actor_train_infos, critic_train_info,
        )

        print(
            "平均每步奖励为 {}.".format(
                critic_train_info[
                    "average_step_rewards"
                ],
            )
        )

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(
                self.done_episodes_rewards,
            )
            print(
                "部分回合已完成，"
                "平均回合奖励为 {}.\n".format(
                    aver_episode_rewards,
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []

    def eval_init(self):
        """初始化评估日志。"""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"][
                "n_rollout_threads"
            ]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        n_eval_threads = (
            self.algo_args["eval"][
                "n_eval_rollout_threads"
            ]
        )
        for eval_i in range(n_eval_threads):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(
        self,
        eval_data,
    ):
        """
        每步记录评估信息。

        参数:
            eval_data: 评估数据元组。
        """
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        n_eval_threads = (
            self.algo_args["eval"][
                "n_eval_rollout_threads"
            ]
        )
        for eval_i in range(n_eval_threads):
            self.one_episode_rewards[eval_i].append(
                eval_rewards[eval_i],
            )
        self.eval_infos = eval_infos

    def eval_thread_done(
        self,
        tid,
    ):
        """
        评估线程完成时记录信息。

        参数:
            tid: 线程编号。
        """
        self.eval_episode_rewards[tid].append(
            np.sum(
                self.one_episode_rewards[tid], axis=0,
            ),
        )
        self.one_episode_rewards[tid] = []

    def eval_log(
        self,
        eval_episode,
    ):
        """
        记录评估日志。

        参数:
            eval_episode: 评估回合数。
        """
        self.eval_episode_rewards = np.concatenate(
            [
                rewards
                for rewards in self.eval_episode_rewards
                if rewards
            ],
        )
        eval_env_infos = {
            "eval_average_episode_rewards": (
                self.eval_episode_rewards
            ),
            "eval_max_episode_rewards": [
                np.max(self.eval_episode_rewards),
            ],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(
            self.eval_episode_rewards,
        )
        print(
            "评估平均回合奖励为 {}.\n".format(
                eval_avg_rew,
            )
        )
        self.log_file.write(
            ",".join(
                map(
                    str,
                    [self.total_num_steps, eval_avg_rew],
                ),
            )
            + "\n"
        )
        self.log_file.flush()

    def log_train(
        self,
        actor_train_infos,
        critic_train_info,
    ):
        """
        记录训练信息。

        参数:
            actor_train_infos: Actor 训练信息列表。
            critic_train_info: Critic 训练信息字典。
        """
        # 记录 Actor 信息
        for agent_id in range(self.num_agents):
            for k, v in (
                actor_train_infos[agent_id].items()
            ):
                agent_k = (
                    "agent%i/" % agent_id + k
                )
                self.writter.add_scalars(
                    agent_k,
                    {agent_k: v},
                    self.total_num_steps,
                )
        # 记录 Critic 信息
        for k, v in critic_train_info.items():
            critic_k = "critic/" + k
            self.writter.add_scalars(
                critic_k,
                {critic_k: v},
                self.total_num_steps,
            )

    def log_env(
        self,
        env_infos,
    ):
        """
        记录环境信息。

        参数:
            env_infos: 环境信息字典。
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(
                    k,
                    {k: np.mean(v)},
                    self.total_num_steps,
                )

    def close(self):
        """关闭日志记录器。"""
        self.log_file.close()
