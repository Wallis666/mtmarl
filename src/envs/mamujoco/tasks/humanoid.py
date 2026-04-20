"""
Humanoid 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供多种运动任务的自定义奖励函数，支持在任务间动态切换。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco.mujoco_multi import (
    MultiAgentMujocoEnv,
)
from numpy.typing import NDArray

from src.utils.reward import tolerance


# ------------------------------------------------------------------
# 任务参数配置
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """正向跑任务参数。"""

    # 目标速度（m/s）
    speed: float = 6.0
    # 躯干最小高度（米），低于此高度视为跌倒
    min_torso_z: float = 0.8
    # 躯干最大高度（米），超出视为异常
    max_torso_z: float = 2.0


@dataclass(frozen=True)
class StandConfig:
    """站立任务参数。"""

    # 理想站立高度（米），指 torso 的 z 坐标
    target_z: float = 1.4
    # 高度容许裕量（米），用于计算 height 奖励的 margin
    z_margin: float = 0.4
    # 允许的最大水平速度（m/s），超出则 slow 奖励衰减
    max_speed: float = 0.5
    # 躯干最小高度（米），低于此高度判定跌倒并终止
    min_torso_z: float = 0.7


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 目标速度（m/s），行走速度较低
    speed: float = 3.0
    # 理想站立高度（米）
    target_z: float = 1.3
    # 高度容许裕量（米）
    z_margin: float = 0.3
    # 躯干最小高度（米），低于此高度判定跌倒
    min_torso_z: float = 0.8


# 全局默认配置实例
_RUN = RunConfig()
_STAND = StandConfig()
_WALK = WalkConfig()


class HumanoidMultiTask(MultiAgentMujocoEnv):
    """
    Humanoid 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Humanoid，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    支持的任务集:
        - run_fwd: 正向跑
        - stand: 站立保持平衡
        - walk: 缓慢行走
    """

    TASKS: list[str] = [
        "run_fwd",
        "stand",
        "walk",
    ]

    # 需要跌倒早期终止的任务集合
    _FALL_TASKS: frozenset[str] = frozenset({
        "run_fwd",
        "stand",
        "walk",
    })

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Humanoid 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "9|8" 表示
                上半身 9 个关节、下半身 8 个关节。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        super().__init__(
            scenario="Humanoid",
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            render_mode=render_mode,
            **kwargs,
        )

        self._render_mode = render_mode
        self._task_idx: int = 0

    # ------------------------------------------------------------------
    # 任务属性
    # ------------------------------------------------------------------

    @property
    def task(self) -> str:
        """返回当前任务名称。"""
        return self.TASKS[self._task_idx]

    @property
    def task_idx(self) -> int:
        """返回当前任务索引。"""
        return self._task_idx

    @property
    def n_tasks(self) -> int:
        """返回支持的任务总数。"""
        return len(self.TASKS)

    # ------------------------------------------------------------------
    # 任务切换
    # ------------------------------------------------------------------

    def set_task(
        self,
        task: str | int,
    ) -> None:
        """
        切换当前任务。

        参数:
            task: 任务名称（字符串）或任务索引（整数）。

        异常:
            ValueError: 当任务名称不在支持列表中时抛出。
            IndexError: 当任务索引超出范围时抛出。
        """
        if isinstance(task, str):
            if task not in self.TASKS:
                raise ValueError(
                    f"不支持的任务: {task!r}，"
                    f"可选任务: {self.TASKS}"
                )
            self._task_idx = self.TASKS.index(task)
        else:
            if not 0 <= task < len(self.TASKS):
                raise IndexError(
                    f"任务索引 {task} 超出范围"
                    f" [0, {len(self.TASKS)})"
                )
            self._task_idx = int(task)

    # ------------------------------------------------------------------
    # 重写 step：替换奖励
    # ------------------------------------------------------------------

    def step(
        self,
        actions: dict[str, NDArray],
    ) -> tuple[
        dict[str, NDArray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """
        执行一步交互，并用当前任务的奖励替换默认奖励。

        参数:
            actions: 各智能体动作的字典，键为智能体名称。

        返回:
            (观测, 奖励, 终止, 截断, 信息) 五元组。
        """
        obs, _, terms, truncs, infos = super().step(actions)
        task_reward = self._compute_reward(infos)
        rewards = {agent: task_reward for agent in obs}
        # 仅在 human 渲染模式下打印，不影响训练
        if self._render_mode == "human":
            vx = self._get_x_velocity(infos)
            torso_z = self._get_torso_z()
            print(
                f"\rtask={self.task:<10} v_x={vx:+6.2f}  "
                f"torso_z={torso_z:.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

        # 跌倒早期终止: 躯干高度低于阈值时结束 episode
        if self._has_fallen():
            terms = {agent: True for agent in terms}

        return obs, rewards, terms, truncs, infos

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _get_x_velocity(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        从信息字典中提取 x 方向速度。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            x 方向线速度。
        """
        # 所有智能体共享同一底层环境，取任意一个即可
        info = next(iter(infos.values()))
        return float(info.get("x_velocity", 0.0))

    def _get_torso_z(self) -> float:
        """
        获取躯干（torso）的 z 轴高度。

        返回:
            torso 当前的 z 坐标值。
        """
        return float(
            self.single_agent_env.unwrapped.data.body(
                "torso"
            ).xpos[2]
        )

    def _get_torso_upright(self) -> float:
        """
        获取躯干的竖直程度。

        通过四元数计算躯干 z 轴在世界坐标系中的投影，
        值为 1 表示完全竖直，0 表示水平，-1 表示倒立。

        返回:
            [-1, 1] 区间内的竖直程度值。
        """
        # root 自由关节: qpos[3:7] 为四元数 (w, x, y, z)
        quat = self.single_agent_env.unwrapped.data.qpos[3:7]
        # 四元数旋转 z 轴单位向量后取 z 分量
        # R(q) * [0,0,1] 的 z 分量 = 1 - 2*(qx^2 + qy^2)
        w, qx, qy, qz = quat
        upright = 1.0 - 2.0 * (qx ** 2 + qy ** 2)
        return float(upright)

    def _has_fallen(self) -> bool:
        """
        判断是否已跌倒。

        对 _FALL_TASKS 中的任务生效，根据各任务的
        最低躯干高度阈值判定。

        返回:
            True 表示应提前终止 episode。
        """
        task = self.task
        if task not in self._FALL_TASKS:
            return False

        torso_z = self._get_torso_z()
        if task == "run_fwd":
            return torso_z < _RUN.min_torso_z
        elif task == "stand":
            return torso_z < _STAND.min_torso_z
        elif task == "walk":
            return torso_z < _WALK.min_torso_z
        return False

    # ------------------------------------------------------------------
    # 奖励分发
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        根据当前任务计算奖励。

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            当前任务对应的标量奖励值。

        异常:
            NotImplementedError: 当前任务未实现时抛出。
        """
        task = self.task
        if task == "run_fwd":
            return self._run_fwd_reward(infos)
        elif task == "stand":
            return self._stand_reward(infos)
        elif task == "walk":
            return self._walk_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _upright_reward(self) -> float:
        """
        躯干竖直姿态奖励。

        基于四元数计算的竖直程度，完全竖直时返回 1，
        水平时返回 0。

        返回:
            [0, 1] 区间内的奖励值。
        """
        upright = self._get_torso_upright()
        return tolerance(
            upright,
            bounds=(0.9, float("inf")),
            margin=1.9,
            sigmoid="linear",
            value_at_margin=0,
        )

    def _height_reward(
        self,
        target_z: float,
        z_margin: float,
    ) -> float:
        """
        躯干高度子奖励。

        鼓励躯干保持在目标高度附近。

        参数:
            target_z: 目标高度（米）。
            z_margin: 高度容许裕量（米）。

        返回:
            [0, 1] 区间内的奖励值。
        """
        torso_z = self._get_torso_z()
        return tolerance(
            torso_z,
            bounds=(target_z, float("inf")),
            margin=z_margin,
            sigmoid="linear",
            value_at_margin=0,
        )

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向跑奖励。

        综合三个子奖励:
            - speed: 沿 x 正方向达到目标速度
            - upright: 躯干保持竖直
            - height: 躯干高度在合理范围内

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        speed_reward = tolerance(
            vx,
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        height = tolerance(
            self._get_torso_z(),
            bounds=(_RUN.min_torso_z, _RUN.max_torso_z),
        )
        return speed_reward * self._upright_reward() * height

    def _stand_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        站立奖励。

        综合三个子奖励:
            - height: 躯干保持在目标站立高度
            - upright: 躯干保持竖直
            - slow: 水平速度接近零，鼓励静止

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        height = self._height_reward(
            _STAND.target_z, _STAND.z_margin,
        )

        vx = self._get_x_velocity(infos)
        slow = tolerance(
            vx,
            bounds=(
                -_STAND.max_speed,
                _STAND.max_speed,
            ),
            sigmoid="linear",
            value_at_margin=0,
        )

        return height * self._upright_reward() * slow

    def _walk_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        行走奖励。

        综合三个子奖励:
            - speed: 沿 x 正方向达到行走目标速度
            - upright: 躯干保持竖直
            - height: 躯干保持在合理站立高度

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        speed_reward = tolerance(
            vx,
            bounds=(_WALK.speed, float("inf")),
            margin=_WALK.speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        height = self._height_reward(
            _WALK.target_z, _WALK.z_margin,
        )

        return speed_reward * self._upright_reward() * height
