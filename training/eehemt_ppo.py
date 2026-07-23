import argparse
import os

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from env.eehemt_env import EEHEMTEnv_Measure_VDS
from env.objectives import NRMSEMinimizeObjective
from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve
from training.ppo_common import build_base_ppo_config
from utils.callbacks import TrainingMetricsCallback

EEHEMT_WANDB_PROJECT = "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"


def add_env_args(
    parser: argparse.ArgumentParser,
    current_dir: str,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--va_file_path",
        type=str,
        default=os.path.join(current_dir, os.getenv("VA_FILE_PATH", "")),
    )
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
    return build_base_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        env_cls=EEHEMTEnv_Measure_VDS,
        env_config=build_env_config(args),
        callbacks_class=TrainingMetricsCallback,
        custom_evaluation_function=evaluate_and_plot_iv_curve,
    )


def build_checkpoint_config() -> tune.CheckpointConfig:
    return tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute=NRMSEMinimizeObjective.RANKED_METRIC,
        checkpoint_score_order=NRMSEMinimizeObjective.RANKED_ORDER,
    )
