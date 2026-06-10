from types import SimpleNamespace

from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve
from train_ppo_tune import (
    build_arg_parser,
    build_checkpoint_config,
    build_env_config,
    build_ppo_config,
)
from utils.callbacks import TrainingMetricsCallback


def test_build_env_config_includes_ir_drop_parameters():
    args = SimpleNamespace(
        va_file_path="/tmp/model.va",
        csv_file_path="/tmp/data.csv",
        random_init=True,
        reduce_obs_err_dim=False,
        reward_norm=True,
        use_stagnation=False,
        rs_ext=1.25,
        rd_ext=0.4,
        ir_drop_n_iter=3,
        ir_drop_maxfev=123,
        nrmse_threshold=7.5,
    )

    env_config = build_env_config(args)

    assert env_config == {
        "va_file_path": "/tmp/model.va",
        "csv_file_path": "/tmp/data.csv",
        "random_init": True,
        "reduce_obs_err_dim": False,
        "reward_norm": True,
        "use_stagnation": False,
        "rs_ext": 1.25,
        "rd_ext": 0.4,
        "ir_drop_n_iter": 3,
        "ir_drop_maxfev": 123,
        "nrmse_threshold": 7.5,
    }


def test_n_iterations_script_default_is_not_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("N_ITERATIONS", "50")
    parser = build_arg_parser("/project")

    args = parser.parse_args([])

    assert args.n_iterations == 100


def test_explicit_n_iterations_argument_has_highest_priority(monkeypatch):
    monkeypatch.setenv("N_ITERATIONS", "50")
    parser = build_arg_parser("/project")

    args = parser.parse_args(["--n_iterations", "123"])

    assert args.n_iterations == 123


def test_explicit_training_argument_has_highest_priority_over_environment(
    monkeypatch,
):
    monkeypatch.setenv("LR", "0.001")
    parser = build_arg_parser("/project")

    args = parser.parse_args(["--lr", "0.002"])

    assert args.lr == 0.002


def test_checkpoint_config_ranks_by_lowest_nrmse():
    checkpoint_config = build_checkpoint_config()

    assert checkpoint_config.checkpoint_score_attribute == "env_runners/min_nrmse"
    assert checkpoint_config.checkpoint_score_order == "min"


def test_build_ppo_config_wires_callbacks_and_driver_evaluation():
    args = SimpleNamespace(
        va_file_path="/tmp/model.va",
        csv_file_path="/tmp/data.csv",
        random_init=True,
        reduce_obs_err_dim=False,
        reward_norm=True,
        use_stagnation=False,
        rs_ext=1.25,
        rd_ext=0.4,
        ir_drop_n_iter=3,
        ir_drop_maxfev=123,
        nrmse_threshold=7.5,
        num_env_runners=2,
        observation_filter="NoFilter",
        train_batch_size_per_learner=128,
        num_epochs=2,
        minibatch_size=64,
        lr=1e-5,
        entropy_coeff=0.01,
        grad_clip=1.0,
        vf_loss_coeff=0.1,
        evaluation_interval=3,
        evaluation_num_env_runners=1,
    )

    config = build_ppo_config(args, num_learners=1, num_gpus_per_learner=0.0)

    assert config.callbacks_class is TrainingMetricsCallback
    assert config.custom_evaluation_function is evaluate_and_plot_iv_curve
    assert config.evaluation_interval == 3
    assert config.evaluation_num_env_runners == 1
    assert config.evaluation_duration == 1
    assert config.evaluation_duration_unit == "episodes"
    assert config.evaluation_config["explore"] is False
    assert config.num_env_runners == 2
    assert config.observation_filter == "NoFilter"
    assert config.train_batch_size_per_learner == 128
    assert config.lr == 1e-5


def test_train_ppo_script_requests_600_iterations():
    with open("scripts/train_ppo.sh", encoding="utf-8") as script_file:
        script = script_file.read()

    assert "--n_iterations 600" in script
    assert "--random_init" in script
    assert "--reduce_obs_err_dim" in script
