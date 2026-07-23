import argparse
import os

import torch as th
from ray.air.integrations.wandb import WandbLoggerCallback

from problems import registry as problem_registry


def build_common_arg_parser(current_dir):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--env", choices=problem_registry.names(), default="hardness"
    )
    parser.add_argument(
        "--observation_filter",
        choices=["NoFilter", "MeanStdFilter"],
        default=os.getenv("OBSERVATION_FILTER", "NoFilter"),
    )
    parser.add_argument(
        "--num_env_runners", type=int, default=int(os.getenv("NUM_ENV_RUNNERS", 4))
    )
    parser.add_argument(
        "--train_batch_size_per_learner",
        type=int,
        default=int(os.getenv("TRAIN_BATCH_SIZE_PER_LEARNER", 4096)),
    )
    parser.add_argument(
        "--num_epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 5))
    )
    parser.add_argument(
        "--minibatch_size", type=int, default=int(os.getenv("MINIBATCH_SIZE", 512))
    )
    parser.add_argument("--lr", type=float, default=float(os.getenv("LR", 5e-6)))
    parser.add_argument(
        "--entropy_coeff", type=float, default=float(os.getenv("ENTROPY_COEFF", 5e-3))
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--vf_loss_coeff", type=float, default=float(os.getenv("VF_LOSS_COEFF", 0.1))
    )
    parser.add_argument(
        "--n_iterations", type=int, default=int(os.getenv("N_ITERATIONS", 100))
    )
    parser.add_argument("--restore_path", type=str, default=os.getenv("RESTORE_PATH", ""))
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        default=int(os.getenv("EVALUATION_INTERVAL", 2)),
    )
    parser.add_argument(
        "--evaluation_num_env_runners",
        type=int,
        default=int(os.getenv("EVALUATION_NUM_ENV_RUNNERS", 1)),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(current_dir, os.getenv("CHECKPOINT_DIR", "")),
    )
    parser.add_argument(
        "--wandb_api_key", type=str, default=os.getenv("WANDB_API_KEY", "")
    )
    return parser


def resolve_learner_resources():
    device_count = th.cuda.device_count()
    if device_count > 0:
        return max(1, device_count // 2), 1.0
    return int(os.getenv("NUM_LEARNERS", 1)), 0.0


def build_wandb_callback(args, *, project_name):
    return WandbLoggerCallback(project=project_name, api_key=args.wandb_api_key)
