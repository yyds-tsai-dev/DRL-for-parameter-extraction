import argparse
import os

import torch as th
from dotenv import load_dotenv
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPOConfig

from env.eehemt_env import EEHEMTEnv_Measure_VDS

# from utils.callbacks import CustomEvalCallbacks
from utils.plot import PlotCurve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    load_dotenv()

    # === Env arguments ===
    current_dir = os.getcwd()
    parser.add_argument(
        "--va_file_path",
        type=str,
        default=os.path.join(current_dir, os.getenv("VA_FILE_PATH", "")),
    )
    # parser.add_argument("--change_param_names", type=str, default=os.getenv("CHANGE_PARAM_NAMES", "UGW,NOF"))
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
    ### New
    parser.add_argument("--vds", type=float, default=float(os.getenv("VDS", 0.5)))
    # parser.add_argument("--test_modified", action="store_true")
    parser.add_argument("--reward_norm", action="store_true")
    parser.add_argument("--use_stagnation", action="store_true")
    parser.add_argument("--reduce_obs_err_dim", action="store_true")

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
        "--n_iterations", type=int, default=int(os.getenv("N_ITERATIONS", 100))
    )  # 100 -> 50, 幾個 sample-train period
    parser.add_argument(
        "--episode_reward_mean",
        type=float,
        default=float(os.getenv("EPISODE_REWARD_MEAN", 5.0)),
    )  # The mean reward to stop training

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

    args = parser.parse_args()

    # === Algo Configure ===
    config = (
        PPOConfig()
        .environment(
            env=EEHEMTEnv_Measure_VDS,
            env_config={
                "va_file_path": args.va_file_path,
                # "simulate_target_data": args.simulate_target_data,
                "csv_file_path": args.csv_file_path,
                # "vds": args.vds,
                # "test_modified": args.test_modified,
                "reduce_obs_err_dim": args.reduce_obs_err_dim,
                "reward_norm": args.reward_norm,
                # "use_stagnation": args.use_stagnation,
            },
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            observation_filter="MeanStdFilter",  # Z-score norm better than L2 norm.
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr * num_learners,
            ### New
            entropy_coeff=args.entropy_coeff,  # type: ignore
            grad_clip=args.grad_clip,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        # .callbacks(CustomEvalCallbacks)
        .callbacks(
            callbacks_class=PlotCurve,
        )
        .evaluation(
            # We only need one evaluation worker for plotting
            evaluation_interval=args.evaluation_interval,
            evaluation_num_env_runners=args.evaluation_num_env_runners,
            evaluation_duration=1,  # Only one episode for evaluation
            evaluation_duration_unit="episodes",
            # custom_evaluation_function=eval_func,
            evaluation_config={"explore": False},
        )
    )

    # tune_config = tune.TuneConfig(
    # metric="episode_reward_mean",
    # mode="max",
    #     reuse_actors=True,
    # )

    # algo_name = os.getenv("ALGO_NAME", "ppo")
    checkpoint_dir = os.path.join(current_dir, os.getenv("CHECKPOINT_DIR", ""))
    stopping_criteria = {"training_iteration": args.n_iterations}
    ckpt_config = tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_score_order="max",
    )
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
        "PPO",
        param_space=config,
        run_config=run_config,
    )
    results = tuner.fit()
    print("\n==== Training completed. ====")

    # === Save model ===
    print(f"\n==== Final algorithm checkpoint saved to: {checkpoint_dir} ====")
