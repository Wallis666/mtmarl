"""多任务环境注册表模块。"""

from src.envs.mamujoco.tasks.cheetah import HalfCheetahMultiTask


ENV_REGISTRY = {
    "2_Agent_Cheetah": HalfCheetahMultiTask,
    "6_Agent_Cheetah": HalfCheetahMultiTask,
}

ARGS_REGISTRY = {
    "2_Agent_Cheetah": {
        "agent_conf": "2x3",
        "agent_obsk": 0,
    },
    "6_Agent_Cheetah": {
        "agent_conf": "6x1",
        "agent_obsk": 0,
    },
}
