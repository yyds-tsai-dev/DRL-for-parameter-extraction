import argparse
import os

import ray
import torch as th
from dotenv import load_dotenv
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.ppo import PPO, PPOConfig

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
    parser.add_argument("--vf_loss_coeff", type=float, default=float(os.getenv("VF_LOSS_COEFF", 0.1)))
    parser.add_argument(
        "--n_iterations", type=int, default=int(os.getenv("N_ITERATIONS", 100))
    )  # 100 -> 50, 幾個 sample-train period
    parser.add_argument(
        "--restore_path", 
        type=str, 
        default=os.getenv("RESTORE_PATH", ""),
        help="Path to the experiment directory to restore from"
    )
    # parser.add_argument(
    #     "--episode_reward_mean",
    #     type=float,
    #     default=float(os.getenv("EPISODE_REWARD_MEAN", 5.0)),
    # )  # The mean reward to stop training

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
                "random_init": args.random_init,
                "reduce_obs_err_dim": args.reduce_obs_err_dim,
                "reward_norm": args.reward_norm,
                "use_stagnation": args.use_stagnation,
            },
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

    checkpoint_dir = os.path.join(current_dir, os.getenv("CHECKPOINT_DIR", ""))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
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
        print(f"\n==== Restoring training from: {args.restore_path} ====")
        
        tuner = tune.Tuner.restore(
            path=args.restore_path,
            trainable=PPO,
            resume_unfinished=True,
            resume_errored=True,
            param_space=config,
        )
    else:
        print("\n==== Starting a NEW training run ====")
        stopping_criteria = {"training_iteration": args.n_iterations}
        ckpt_config = tune.CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute="env_runners/episode_return_mean",
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
            print(
                "\n==== Training stopped before reaching the requested "
                f"{args.n_iterations} iterations. ===="
            )
            raise SystemExit(130)
    print("\n==== Training completed. ====")

    # === Save model ===
    print(f"\n==== Final algorithm checkpoint saved to: {checkpoint_dir} ====")
