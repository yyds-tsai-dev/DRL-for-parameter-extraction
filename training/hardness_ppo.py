import argparse
import os

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from env.material_hardness_env import MaterialHardnessEnv
from evaluation.hardness_evaluation import evaluate_and_save_hardness
from utils.hardness_callbacks import HardnessMetricsCallback

HARDNESS_WANDB_PROJECT = "PPO_for_material_hardness_optimization"


def add_env_args(
    parser: argparse.ArgumentParser,
    current_dir: str,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--hardness_model_package_path",
        type=str,
        default=os.path.join(
            current_dir,
            os.getenv(
                "HARDNESS_MODEL_PACKAGE_PATH",
                "env/hardness/XGB_model_selection_package.zip",
            ),
        ),
    )
    parser.add_argument(
        "--hardness_threshold",
        type=float,
        default=float(os.getenv("HARDNESS_THRESHOLD", 650.0)),
    )
    parser.add_argument(
        "--hardness_reward_scale",
        type=float,
        default=float(os.getenv("HARDNESS_REWARD_SCALE", 100.0)),
    )
    parser.add_argument(
        "--hardness_reward_min",
        type=float,
        default=float(os.getenv("HARDNESS_REWARD_MIN", -3.0)),
    )
    parser.add_argument(
        "--hardness_reward_max",
        type=float,
        default=float(os.getenv("HARDNESS_REWARD_MAX", 3.0)),
    )
    return parser


def build_env_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_package_path": args.hardness_model_package_path,
        "hardness_threshold": args.hardness_threshold,
        "reward_scale": args.hardness_reward_scale,
        "reward_min": args.hardness_reward_min,
        "reward_max": args.hardness_reward_max,
    }


def build_ppo_config(
    args: argparse.Namespace,
    *,
    num_learners: int,
    num_gpus_per_learner: float,
) -> PPOConfig:
    from training.ppo_common import build_base_ppo_config

    return build_base_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        env_cls=MaterialHardnessEnv,
        env_config=build_env_config(args),
        callbacks_class=HardnessMetricsCallback,
        custom_evaluation_function=evaluate_and_save_hardness,
    )


def build_checkpoint_config() -> tune.CheckpointConfig:
    return tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="env_runners/max_predicted_hardness",
        checkpoint_score_order="max",
    )
