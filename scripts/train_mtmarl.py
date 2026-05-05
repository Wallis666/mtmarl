"""
MTMARL 训练入口脚本。

用法:
    python -m scripts.train_mtmarl
    python -m scripts.train_mtmarl --config path/to/config.yaml
"""

import argparse

import yaml

from src.runners.world_model_runner import WorldModelRunner


def main():
    """解析参数并启动训练。"""
    parser = argparse.ArgumentParser(
        description="MTMARL 训练",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mamujoco/mtmarl.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="实验名称",
    )
    parsed = parser.parse_args()

    with open(
        parsed.config, "r", encoding="utf-8",
    ) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    env_args = cfg.pop("env")
    algo_args = cfg

    args = {
        "algo": "mtmarl",
        "env": "mamujoco",
        "exp_name": parsed.exp_name,
    }

    runner = WorldModelRunner(args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
