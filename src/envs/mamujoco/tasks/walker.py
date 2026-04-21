"""
Walker2d 多任务多智能体环境模块。

基于 gymnasium_robotics MaMuJoCo 的 MultiAgentMujocoEnv 派生，
提供 stand、walk_fwd、walk_bwd、run_fwd、run_bwd 五种任务的
自定义奖励函数，支持在任务间动态切换。
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
class CommonConfig:
    """各任务共用参数。"""

    # 站立时躯干最低高度（米），用于站立基底子奖励
    stand_height: float = 1.2


@dataclass(frozen=True)
class WalkConfig:
    """行走任务参数。"""

    # 目标行走速度（m/s）
    speed: float = 1.5


@dataclass(frozen=True)
class RunConfig:
    """奔跑任务参数。"""

    # 目标奔跑速度（m/s）
    speed: float = 6.0


@dataclass(frozen=True)
class GaitConfig:
    """步态交替参数。"""

    # cfrc_ext 力范数阈值，超过此值判定为着地
    contact_threshold: float = 1.0
    # EMA 衰减率，越大越关注近期接触历史
    ema_alpha: float = 0.1


# 全局默认配置实例
_COMMON = CommonConfig()
_WALK = WalkConfig()
_RUN = RunConfig()
_GAIT = GaitConfig()


class Walker2dMultiTask(MultiAgentMujocoEnv):
    """
    Walker2d 多任务多智能体环境。

    继承 MultiAgentMujocoEnv，固定 scenario 为 Walker2d，
    并在 step 中用当前任务的自定义奖励替换默认奖励。
    各智能体在同一时间步共享相同的任务奖励信号。

    奖励设计以站立基底子奖励（高度 + 竖直）为核心，
    运动任务在此基础上乘以速度因子和步态交替因子。
    步态因子通过左右脚接触的指数移动平均跟踪双脚
    使用对称性，防止智能体退化为单脚跳。

    支持的任务集:
        - stand: 站立保持平衡
        - walk_fwd: 正向行走
        - walk_bwd: 反向行走
        - run_fwd: 正向奔跑
        - run_bwd: 反向奔跑
    """

    TASKS: list[str] = [
        "stand",
        "walk_fwd",
        "walk_bwd",
        "run_fwd",
        "run_bwd",
    ]

    def __init__(
        self,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        初始化 Walker2d 多任务环境。

        参数:
            agent_conf: 智能体分割配置，如 "2x3" 表示
                2 个智能体各控制 3 个关节。
            agent_obsk: 观测深度，0 为仅局部，1 为局部加
                一阶邻居。
            render_mode: 渲染模式，如 "human" 或
                "rgb_array"。
            **kwargs: 传递给 MultiAgentMujocoEnv 的额外
                参数。
        """
        super().__init__(
            scenario="Walker2d",
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            render_mode=render_mode,
            **kwargs,
        )

        self._render_mode = render_mode
        self._task_idx: int = 0

        # 缓存左右脚 body id，避免每步查找
        env = self.single_agent_env.unwrapped
        self._foot_id: int = env.model.body(
            "foot"
        ).id
        self._foot_left_id: int = env.model.body(
            "foot_left"
        ).id

        # 步态追踪: 左右脚接触的指数移动平均，
        # 初始 0.5 表示双脚均匀使用（中性假设）
        self._foot_ema: NDArray = np.array(
            [0.5, 0.5]
        )

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
    # 重写 reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, NDArray], dict[str, dict]]:
        """
        重置环境并初始化步态追踪状态。

        参数:
            seed: 随机种子。
            options: 重置选项。

        返回:
            (观测, 信息) 二元组。
        """
        result = super().reset(
            seed=seed, options=options,
        )
        self._foot_ema = np.array([0.5, 0.5])
        return result

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

        # 更新步态追踪（在计算奖励之前）
        self._update_foot_ema()

        task_reward = self._compute_reward(infos)
        rewards = {agent: task_reward for agent in obs}
        # 仅在 human 渲染模式下打印，不影响训练
        if self._render_mode == "human":
            vx = self._get_x_velocity(infos)
            height = self._get_torso_height()
            upright = self._get_torso_upright()
            gait = self._gait_reward()
            print(
                f"\rtask={self.task:<10} "
                f"v_x={vx:+6.2f}  "
                f"height={height:.2f}  "
                f"upright={upright:+.2f}  "
                f"gait={gait:.2f}  "
                f"r={task_reward:.3f} ",
                end="",
                flush=True,
            )

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

    def _get_torso_height(self) -> float:
        """
        获取躯干高度（rootz 关节位置）。

        Walker2d 的 qpos[1] 即 rootz 滑动关节，
        表示躯干在世界坐标系中的 z 位置。

        返回:
            躯干 z 坐标（米）。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[1]
        )

    def _get_torso_pitch(self) -> float:
        """
        获取躯干俯仰角（rooty 关节位置）。

        返回:
            躯干俯仰角（弧度），0 表示竖直，
            正值为前倾，负值为后仰。
        """
        return float(
            self.single_agent_env.unwrapped.data.qpos[2]
        )

    def _get_torso_upright(self) -> float:
        """
        获取躯干的竖直程度。

        通过 cos(pitch) 计算，等效于 torso 旋转矩阵
        的 zz 分量: 1 表示完全竖直，0 表示水平，
        -1 表示倒立。

        返回:
            [-1, 1] 区间内的竖直程度值。
        """
        return float(np.cos(self._get_torso_pitch()))

    # ------------------------------------------------------------------
    # 步态追踪
    # ------------------------------------------------------------------

    def _get_foot_contacts(self) -> tuple[bool, bool]:
        """
        检测左右脚是否着地。

        通过 cfrc_ext 的力范数判定接触:
        范数超过阈值视为该脚正在承受地面反力。

        返回:
            (右脚着地, 左脚着地) 布尔元组。
        """
        ext = self.single_agent_env.unwrapped.data.cfrc_ext
        right = float(np.linalg.norm(
            ext[self._foot_id]
        ))
        left = float(np.linalg.norm(
            ext[self._foot_left_id]
        ))
        th = _GAIT.contact_threshold
        return right > th, left > th

    def _update_foot_ema(self) -> None:
        """
        更新左右脚接触的指数移动平均。

        每步将当前接触状态（0/1）混入 EMA，
        跟踪近期各脚的使用频率。
        """
        right, left = self._get_foot_contacts()
        a = _GAIT.ema_alpha
        self._foot_ema[0] = (
            (1 - a) * self._foot_ema[0]
            + a * float(right)
        )
        self._foot_ema[1] = (
            (1 - a) * self._foot_ema[1]
            + a * float(left)
        )

    def _gait_reward(self) -> float:
        """
        步态交替子奖励。

        计算左右脚 EMA 的对称比:
            min(ema) / max(ema)
        双脚交替使用时接近 1，单脚跳时接近 0。
        双脚均无明显接触（如静止站立初期 EMA
        自然衰减）时返回 1，不施加惩罚。

        返回:
            [0, 1] 区间内的奖励值。
        """
        ema_max = float(max(
            self._foot_ema[0], self._foot_ema[1],
        ))
        if ema_max < 0.05:
            return 1.0
        ema_min = float(min(
            self._foot_ema[0], self._foot_ema[1],
        ))
        return ema_min / ema_max

    # ------------------------------------------------------------------
    # 共用子奖励
    # ------------------------------------------------------------------

    def _standing_reward(self) -> float:
        """
        站立基底子奖励（各任务共用）。

        综合 torso 高度和竖直程度两个信号:
            - standing: torso 高度 >= 1.2m 时满分，
                低于时 gaussian 衰减
            - upright: 竖直程度映射到 [0, 1]
        加权求和: (3 * standing + upright) / 4，
        高度权重更大因其对站立稳定性更关键。

        返回:
            [0, 1] 区间内的奖励值。
        """
        standing = tolerance(
            self._get_torso_height(),
            bounds=(
                _COMMON.stand_height, float("inf"),
            ),
            margin=_COMMON.stand_height / 2,
        )
        upright = (1 + self._get_torso_upright()) / 2
        return (3 * standing + upright) / 4

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
        if task == "stand":
            return self._stand_reward()
        elif task == "walk_fwd":
            return self._walk_fwd_reward(infos)
        elif task == "walk_bwd":
            return self._walk_bwd_reward(infos)
        elif task == "run_fwd":
            return self._run_fwd_reward(infos)
        elif task == "run_bwd":
            return self._run_bwd_reward(infos)
        else:
            raise NotImplementedError(
                f"任务 {task!r} 尚未实现"
            )

    # ------------------------------------------------------------------
    # 各任务奖励函数
    # ------------------------------------------------------------------

    def _stand_reward(self) -> float:
        """
        站立奖励。

        直接使用站立基底子奖励，要求躯干高度
        达标且保持竖直姿态。不乘步态因子，
        因为站立不要求双脚交替。

        返回:
            [0, 1] 区间内的奖励值。
        """
        return self._standing_reward()

    def _walk_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向行走奖励。

        在站立基底上乘以速度因子和步态因子:
            - standing: 高度 + 竖直基底
            - move: 沿 x 正方向达到目标速度，
                压缩到 [1/6, 1] 保证速度不够时
                仍有站立激励
            - gait: 双脚交替使用的对称比，
                防止退化为单脚跳

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(_WALK.speed, float("inf")),
            margin=_WALK.speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return (
            self._standing_reward()
            * (5 * move + 1) / 6
            * self._gait_reward()
        )

    def _walk_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向行走奖励。

        在站立基底上乘以速度因子和步态因子:
            - standing: 高度 + 竖直基底
            - move: 沿 x 负方向达到目标速度
            - gait: 双脚交替使用的对称比

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(-float("inf"), -_WALK.speed),
            margin=_WALK.speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return (
            self._standing_reward()
            * (5 * move + 1) / 6
            * self._gait_reward()
        )

    def _run_fwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        正向奔跑奖励。

        在站立基底上乘以速度因子和步态因子:
            - standing: 高度 + 竖直基底
            - move: 沿 x 正方向达到目标速度
            - gait: 双脚交替使用的对称比

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(_RUN.speed, float("inf")),
            margin=_RUN.speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return (
            self._standing_reward()
            * (5 * move + 1) / 6
            * self._gait_reward()
        )

    def _run_bwd_reward(
        self,
        infos: dict[str, dict],
    ) -> float:
        """
        反向奔跑奖励。

        在站立基底上乘以速度因子和步态因子:
            - standing: 高度 + 竖直基底
            - move: 沿 x 负方向达到目标速度
            - gait: 双脚交替使用的对称比

        参数:
            infos: 环境 step 返回的信息字典。

        返回:
            [0, 1] 区间内的奖励值。
        """
        vx = self._get_x_velocity(infos)
        move = tolerance(
            vx,
            bounds=(-float("inf"), -_RUN.speed),
            margin=_RUN.speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return (
            self._standing_reward()
            * (5 * move + 1) / 6
            * self._gait_reward()
        )
