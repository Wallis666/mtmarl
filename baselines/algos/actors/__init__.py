"""算法注册表模块。"""

from baselines.algos.actors.haa2c import HAA2C
from baselines.algos.actors.had3qn import HAD3QN
from baselines.algos.actors.haddpg import HADDPG
from baselines.algos.actors.happo import HAPPO
from baselines.algos.actors.hasac import HASAC
from baselines.algos.actors.hatd3 import HATD3
from baselines.algos.actors.hatrpo import HATRPO
from baselines.algos.actors.maddpg import MADDPG
from baselines.algos.actors.mappo import MAPPO
from baselines.algos.actors.matd3 import MATD3


ALGO_REGISTRY = {
    "haa2c": HAA2C,
    "had3qn": HAD3QN,
    "haddpg": HADDPG,
    "happo": HAPPO,
    "hasac": HASAC,
    "hatd3": HATD3,
    "hatrpo": HATRPO,
    "maddpg": MADDPG,
    "mappo": MAPPO,
    "matd3": MATD3,
}
