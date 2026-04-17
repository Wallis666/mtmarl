"""评价器注册表模块。"""

from baselines.algos.critics.continuous_q_critic import ContinuousQCritic
from baselines.algos.critics.discrete_q_critic import DiscreteQCritic
from baselines.algos.critics.soft_twin_continuous_q_critic import (
    SoftTwinContinuousQCritic,
)
from baselines.algos.critics.twin_continuous_q_critic import TwinContinuousQCritic
from baselines.algos.critics.v_critic import VCritic


CRITIC_REGISTRY = {
    "haa2c": VCritic,
    "had3qn": DiscreteQCritic,
    "haddpg": ContinuousQCritic,
    "happo": VCritic,
    "hasac": SoftTwinContinuousQCritic,
    "hatd3": TwinContinuousQCritic,
    "hatrpo": VCritic,
    "maddpg": ContinuousQCritic,
    "mappo": VCritic,
    "matd3": TwinContinuousQCritic,
}
