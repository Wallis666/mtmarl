"""运行器注册表模块。"""

from baselines.runners.off_policy_ha_runner import OffPolicyHARunner
from baselines.runners.off_policy_ma_runner import OffPolicyMARunner
from baselines.runners.on_policy_ha_runner import OnPolicyHARunner
from baselines.runners.on_policy_ma_runner import OnPolicyMARunner


RUNNER_REGISTRY = {
    "haa2c": OnPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "happo": OnPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "matd3": OffPolicyMARunner,
}
