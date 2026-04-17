"""环境注册表模块。"""

from baselines.envs.mamujoco.mamujoco_logger import (
    MAMuJoCoLogger,
)


LOGGER_REGISTRY = {
    "mamujoco": MAMuJoCoLogger,
}
