"""基于 OpenAI Baselines 修改，适配多智能体环境。"""

import copy
from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process

import numpy as np


def tile_images(img_nhwc):
    """
    将 N 张图像拼接为一张大的 PxQ 图像。
    (P, Q) 尽可能接近，若 N 为完全平方数则 P=Q。
    输入: img_nhwc，图像列表或数组，转为数组后 ndim=4
        n = 批次索引, h = 高度, w = 宽度, c = 通道
    返回:
        bigim_HWc，ndim=3 的 ndarray
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(N, H * W)]
    )
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudpickleWrapper(object):
    """
    使用 cloudpickle 序列化内容
    （否则 multiprocessing 会尝试使用 pickle）。
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    抽象的异步向量化环境。
    用于将多个环境副本的数据进行批处理，使得每个观测
    变为一批观测，期望的动作为每个环境对应的一批动作。
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs,
        observation_space,
        share_observation_space,
        action_space,
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        重置所有环境并返回观测数组或观测数组字典。

        如果 step_async 仍在执行，该工作将被取消，
        在再次调用 step_async() 之前不应调用
        step_wait()。
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        通知所有环境使用给定动作开始执行一步。
        调用 step_wait() 获取该步的结果。

        如果已有 step_async 正在执行，则不应调用此方法。
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        等待 step_async() 发起的步骤完成。

        返回 (obs, rews, dones, infos):
         - obs: 观测数组或观测数组字典
         - rews: 奖励数组
         - dones: "回合结束" 布尔值数组
         - infos: 信息对象序列
        """
        pass

    def close_extras(self):
        """
        清理基类之外的额外资源。
        仅在 self.closed 为 False 时运行。
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        同步地执行环境步骤。

        保留此方法用于向后兼容。
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """从每个环境返回 RGB 图像。"""
        raise NotImplementedError

    @property
    def unwrapped(self):
        """返回底层未包装的环境。"""
        return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def shareworker(
    remote,
    parent_remote,
    env_fn_wrapper,
):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = (
                env.step(data)
            )
            if "bool" in done.__class__.__name__:
                if done:
                    # 回合结束，保存原始obs、state和
                    # available_actions到info，然后重置
                    info[0]["original_obs"] = (
                        copy.deepcopy(ob)
                    )
                    info[0]["original_state"] = (
                        copy.deepcopy(s_ob)
                    )
                    info[0]["original_avail_actions"] = (
                        copy.deepcopy(available_actions)
                    )
                    ob, s_ob, available_actions = (
                        env.reset()
                    )
            else:
                if np.all(done):
                    # 回合结束，保存原始obs、state和
                    # available_actions到info，然后重置
                    info[0]["original_obs"] = (
                        copy.deepcopy(ob)
                    )
                    info[0]["original_state"] = (
                        copy.deepcopy(s_ob)
                    )
                    info[0]["original_avail_actions"] = (
                        copy.deepcopy(available_actions)
                    )
                    ob, s_ob, available_actions = (
                        env.reset()
                    )

            remote.send(
                (ob, s_ob, reward, done, info,
                 available_actions)
            )
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == "reset_task":
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == "set_task":
            env.set_task(data)
            remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((
                env.observation_space,
                env.share_observation_space,
                env.action_space,
            ))
        elif cmd == "render_vulnerability":
            fr = env.render_vulnerability(data)
            remote.send((fr))
        elif cmd == "get_num_agents":
            remote.send((env.n_agents))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    """使用子进程运行的共享向量化环境。"""

    def __init__(
        self,
        env_fns,
        spaces=None,
    ):
        """
        env_fns: 要在子进程中运行的 gym 环境列表。
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(nenvs)]
        )
        self.ps = [
            Process(
                target=shareworker,
                args=(
                    work_remote,
                    remote,
                    CloudpickleWrapper(env_fn),
                ),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes,
                self.remotes,
                env_fns,
            )
        ]
        for p in self.ps:
            # 如果主进程崩溃，不应导致子进程挂起
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        (
            observation_space,
            share_observation_space,
            action_space,
        ) = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            observation_space,
            share_observation_space,
            action_space,
        )

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [
            remote.recv() for remote in self.remotes
        ]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(
            *results
        )
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [
            remote.recv() for remote in self.remotes
        ]
        obs, share_obs, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(available_actions),
        )

    def reset_task(self):
        for remote in self.remotes:
            remote.send(("reset_task", None))
        return np.stack(
            [remote.recv() for remote in self.remotes]
        )

    def meta_set_task(self, task_idxes):
        """为每个环境设置不同的任务。"""
        assert len(task_idxes) == len(self.remotes)
        for remote, task_idx in zip(
            self.remotes, task_idxes,
        ):
            remote.send(("set_task", task_idx))
        for remote in self.remotes:
            remote.recv()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


# 单环境
class ShareDummyVecEnv(ShareVecEnv):
    """单环境的共享虚拟向量化环境。"""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
        except Exception:
            pass

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [
            env.step(a)
            for (a, env) in zip(self.actions, self.envs)
        ]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results)
        )

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:
                if done:
                    # 回合结束，保存原始obs、state和
                    # available_actions到info，然后重置
                    infos[i][0]["original_obs"] = (
                        copy.deepcopy(obs[i])
                    )
                    infos[i][0]["original_state"] = (
                        copy.deepcopy(share_obs[i])
                    )
                    infos[i][0][
                        "original_avail_actions"
                    ] = copy.deepcopy(
                        available_actions[i]
                    )
                    (
                        obs[i],
                        share_obs[i],
                        available_actions[i],
                    ) = self.envs[i].reset()
            else:
                if np.all(done):
                    # 回合结束，保存原始obs、state和
                    # available_actions到info，然后重置
                    infos[i][0]["original_obs"] = (
                        copy.deepcopy(obs[i])
                    )
                    infos[i][0]["original_state"] = (
                        copy.deepcopy(share_obs[i])
                    )
                    infos[i][0][
                        "original_avail_actions"
                    ] = copy.deepcopy(
                        available_actions[i]
                    )
                    (
                        obs[i],
                        share_obs[i],
                        available_actions[i],
                    ) = self.envs[i].reset()
        self.actions = None

        return (
            obs,
            share_obs,
            rews,
            dones,
            infos,
            available_actions,
        )

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(
            np.array, zip(*results)
        )
        return obs, share_obs, available_actions

    def meta_set_task(self, task_idxes):
        """为每个环境设置不同的任务。"""
        assert len(task_idxes) == len(self.envs)
        for env, task_idx in zip(
            self.envs, task_idxes,
        ):
            env.set_task(task_idx)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array(
                [env.render(mode=mode)
                 for env in self.envs]
            )
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError
