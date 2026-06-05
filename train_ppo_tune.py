import argparse
import os

import ray
import torch as th
from dotenv import load_dotenv
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from env.eehemt_env import EEHEMTEnv_Measure_VDS
from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve

# from utils.callbacks import CustomEvalCallbacks
from utils.callbacks import TrainingMetricsCallback
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def build_arg_parser(current_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # === Env arguments ===
    parser.add_argument(
        "--va_file_path",
        type=str,
        default=os.path.join(current_dir, os.getenv("VA_FILE_PATH", "")),
    )
    # parser.add_argument(
    #     "--simulate_target_data",
    #     action="store_true",
    #     help="Whether to simulate target data",
    # )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default=os.path.join(current_dir, os.getenv("CSV_FILE_PATH", "")),
    )
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--reduce_obs_err_dim", action="store_true")
    parser.add_argument("--reward_norm", action="store_true")
    parser.add_argument("--use_stagnation", action="store_true")
    parser.add_argument(
        "--rs_ext",
        type=float,
        default=float(os.getenv("RS_EXT", 0.0)),
        help="External source resistance used by IR-drop correction.",
    )
    parser.add_argument(
        "--rd_ext",
        type=float,
        default=float(os.getenv("RD_EXT", 0.0)),
        help="External drain resistance used by IR-drop correction.",
    )
    parser.add_argument(
        "--ir_drop_n_iter",
        type=int,
        default=int(os.getenv("IR_DROP_N_ITER", 2)),
        help="Fixed-point warmup iterations before IR-drop fsolve.",
    )
    parser.add_argument(
        "--ir_drop_maxfev",
        type=int,
        default=int(os.getenv("IR_DROP_MAXFEV", 40)),
        help="Maximum fsolve evaluations for each IR-drop curve solve.",
    )
    parser.add_argument(
        "--nrmse_threshold",
        type=float,
        default=float(os.getenv("NRMSE_THRESHOLD", 10.0)),
        help="Success threshold for the NRMSE objective, expressed as a percentage.",
    )
    parser.add_argument(
        "--observation_filter",
        choices=["NoFilter", "MeanStdFilter"],
        default=os.getenv("OBSERVATION_FILTER", "NoFilter"),
        help="RLlib observation filter. NoFilter keeps the Gymnasium observation contract intact.",
    )

    # === Env runner arguments ===
    parser.add_argument(
        "--num_env_runners", type=int, default=int(os.getenv("NUM_ENV_RUNNERS", 4))
    )

    # === Training arguments ===
    parser.add_argument(
        "--train_batch_size_per_learner",
        type=int,
        default=int(os.getenv("TRAIN_BATCH_SIZE_PER_LEARNER", 4096)),
    )
    parser.add_argument(
        "--num_epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 5))
    )  # 從 env 收集到的資料重複使用多少次來進行 model 更新
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
        "--n_iterations",
        type=int,
        default=100,
    )  # 幾個 sample-train period；script default wins over .env N_ITERATIONS.
    parser.add_argument(
        "--restore_path",
        type=str,
        default=os.getenv("RESTORE_PATH", ""),
        help="Path to the experiment directory to restore from",
    )
    # parser.add_argument(
    #     "--episode_reward_mean",
    #     type=float,
    #     default=float(os.getenv("EPISODE_REWARD_MEAN", 5.0)),
    # )  # The mean reward to stop training

    # === Evaluation arguments ===
    # parser.add_argument("--log_y", action="store_true")
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
    return parser


def build_env_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "va_file_path": args.va_file_path,
        "csv_file_path": args.csv_file_path,
        "random_init": args.random_init,
        "reduce_obs_err_dim": args.reduce_obs_err_dim,
        "reward_norm": args.reward_norm,
        "use_stagnation": args.use_stagnation,
        "rs_ext": args.rs_ext,
        "rd_ext": args.rd_ext,
        "ir_drop_n_iter": args.ir_drop_n_iter,
        "ir_drop_maxfev": args.ir_drop_maxfev,
        "nrmse_threshold": args.nrmse_threshold,
    }


def build_ppo_config(
    args: argparse.Namespace,
    *,
    num_learners: int,
    num_gpus_per_learner: float,
) -> PPOConfig:
    return (
        PPOConfig()
        .environment(
            env=EEHEMTEnv_Measure_VDS,
            env_config=build_env_config(args),
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            observation_filter=args.observation_filter,
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr * num_learners,
            entropy_coeff=args.entropy_coeff,  # type: ignore
            grad_clip=args.grad_clip,
            # model={
            #     "fcnet_hiddens": [512, 512],
            # "post_fcnet_hiddens": [512],
            # "vf_share_layers": False,
            # }
            vf_loss_coeff=args.vf_loss_coeff,
            vf_clip_param=20.0,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        # .callbacks(CustomEvalCallbacks)
        .callbacks(
            callbacks_class=TrainingMetricsCallback,
        )
        .evaluation(
            # We only need one evaluation worker for plotting
            evaluation_interval=args.evaluation_interval,
            evaluation_num_env_runners=args.evaluation_num_env_runners,
            evaluation_duration=1,  # Only one episode for evaluation
            evaluation_duration_unit="episodes",
            custom_evaluation_function=evaluate_and_plot_iv_curve,
            evaluation_config={"explore": False},
        )
    )


def build_checkpoint_config() -> tune.CheckpointConfig:
    return tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="env_runners/min_nrmse",
        checkpoint_score_order="min",
    )


if __name__ == "__main__":
    load_dotenv()
    configure_logging()

    current_dir = os.getcwd()
    os.environ.setdefault("PROJECT_ROOT", current_dir)
    parser = build_arg_parser(current_dir)

    # === Learner arguments ===
    device_count = th.cuda.device_count()
    if device_count > 0:
        num_learners = device_count // 2
        if num_learners == 0:
            num_learners = 1
        num_gpus_per_learner = 1.0
    else:
        num_learners = int(os.getenv("NUM_LEARNERS", 1))
        num_gpus_per_learner = 0.0

    args = parser.parse_args()

    # === Algo Configure ===
    config = build_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
    )

    # tune_config = tune.TuneConfig(
    # metric="episode_reward_mean",
    # mode="max",
    #     reuse_actors=True,
    # )

    checkpoint_dir = os.path.join(current_dir, os.getenv("CHECKPOINT_DIR", ""))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
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
            ]
        },
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
        stopping_criteria = {"training_iteration": args.n_iterations}
        ckpt_config = build_checkpoint_config()
        run_config = tune.RunConfig(
            name="EEHEMT_PPO",
            storage_path=checkpoint_dir,
            stop=stopping_criteria,
            checkpoint_config=ckpt_config,
            callbacks=[
                WandbLoggerCallback(
                    project="PPO_for_multi_I-V_curves_fitting_in_EEHEMT",
                    api_key=os.getenv("WANDB_API_KEY", default=""),
                )
            ],
        )
        tuner = tune.Tuner(
            PPO,
            param_space=config,
            run_config=run_config,
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

    # === Save model ===
    logger.info("\n==== Final algorithm checkpoint saved to: %s ====", checkpoint_dir)
