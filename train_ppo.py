import argparse
import os
import sys
from types import ModuleType

import ray
from dotenv import load_dotenv
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from problems import registry as problem_registry
from training.ppo_common import (
    build_common_arg_parser,
    build_wandb_callback,
    resolve_learner_resources,
)
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def select_training_module(env_name: str) -> ModuleType:
    return problem_registry.get(env_name).module


def build_arg_parser(
    current_dir: str,
    argv: list[str] | None = None,
) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--env", choices=problem_registry.names(), default="hardness"
    )
    pre_args, _ = pre_parser.parse_known_args([] if argv is None else argv)

    training_module = select_training_module(pre_args.env)
    parser = argparse.ArgumentParser(parents=[build_common_arg_parser(current_dir)])
    training_module.add_env_args(parser, current_dir)
    return parser


def build_ray_runtime_env(current_dir: str) -> dict[str, object]:
    return {
        "env_vars": {
            "PROJECT_ROOT": current_dir,
        },
        "excludes": [
            ".git/**",
            ".venv/**",
            ".mypy_cache/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            ".codebase-memory/**",
            "result/**",
            "demo/demo.tar.gz",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    }


def _wandb_project_name(env_name: str, training_module: ModuleType) -> str:
    del training_module  # kept for call-site stability; registry owns the mapping
    return problem_registry.get(env_name).wandb_project


def main() -> None:
    load_dotenv()
    configure_logging()

    current_dir = os.getcwd()
    os.environ.setdefault("PROJECT_ROOT", current_dir)

    parser = build_arg_parser(current_dir, sys.argv[1:])
    args = parser.parse_args()
    training_module = select_training_module(args.env)

    num_learners, num_gpus_per_learner = resolve_learner_resources()
    config = training_module.build_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
    )

    ray.init(
        ignore_reinit_error=True,
        runtime_env=build_ray_runtime_env(current_dir),
    )
    if args.restore_path:
        logger.info("\n==== Restoring training from: %s ====", args.restore_path)
        tuner = tune.Tuner.restore(
            path=args.restore_path,
            trainable=PPO,
            resume_unfinished=True,
            resume_errored=True,
            param_space=config,
        )
    else:
        logger.info("\n==== Starting a NEW training run ====")
        tuner = tune.Tuner(
            PPO,
            param_space=config,
            run_config=tune.RunConfig(
                name=f"{args.env.upper()}_PPO",
                storage_path=args.checkpoint_dir,
                stop={"training_iteration": args.n_iterations},
                checkpoint_config=training_module.build_checkpoint_config(),
                callbacks=[
                    build_wandb_callback(
                        args,
                        project_name=_wandb_project_name(args.env, training_module),
                    )
                ],
            ),
        )

    results = tuner.fit()
    if not args.restore_path:
        completed_iterations = [
            result.metrics.get("training_iteration", 0)
            for result in results
            if result.metrics
        ]
        if not completed_iterations or max(completed_iterations) < args.n_iterations:
            logger.info(
                "\n==== Training stopped before reaching the requested "
                f"{args.n_iterations} iterations. ===="
            )
            raise SystemExit(130)

    logger.info("\n==== Training completed. ====")
    logger.info("\n==== Final algorithm checkpoint saved to: %s ====", args.checkpoint_dir)


if __name__ == "__main__":
    main()
