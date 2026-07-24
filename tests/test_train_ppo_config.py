from types import SimpleNamespace
from pathlib import Path

import pytest

from env.eehemt_env import EEHEMTEnv_Measure_VDS
from env.material_hardness_env import MaterialHardnessEnv
from evaluation.hardness_evaluation import evaluate_and_save_hardness
from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve
from train_ppo import build_arg_parser, select_training_module
from training import eehemt_ppo, hardness_ppo
from training.ppo_common import build_common_arg_parser
from utils.callbacks import TrainingMetricsCallback
from utils.hardness_callbacks import HardnessMetricsCallback


def _common_ppo_args(**overrides):
    args = {
        "num_env_runners": 2,
        "observation_filter": "NoFilter",
        "train_batch_size_per_learner": 128,
        "num_epochs": 2,
        "minibatch_size": 64,
        "lr": 1e-5,
        "entropy_coeff": 0.01,
        "grad_clip": 1.0,
        "vf_loss_coeff": 0.1,
        "evaluation_interval": 3,
        "evaluation_num_env_runners": 1,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _eehemt_args(**overrides):
    return _common_ppo_args(
        va_file_path="/tmp/model.va",
        csv_file_path="/tmp/data.csv",
        random_init=True,
        reduce_obs_err_dim=False,
        reward_norm=True,
        rs_ext=1.25,
        rd_ext=0.4,
        ir_drop_n_iter=3,
        ir_drop_maxfev=123,
        nrmse_threshold=7.5,
        **overrides,
    )


def _hardness_args(**overrides):
    return _common_ppo_args(
        hardness_model_package_path="/tmp/hardness.zip",
        hardness_threshold=700.0,
        hardness_reward_scale=50.0,
        hardness_reward_min=-2.0,
        hardness_reward_max=4.0,
        **overrides,
    )


def test_eehemt_env_config_includes_ir_drop_parameters():
    env_config = eehemt_ppo.build_env_config(_eehemt_args())

    assert env_config == {
        "va_file_path": "/tmp/model.va",
        "csv_file_path": "/tmp/data.csv",
        "random_init": True,
        "reduce_obs_err_dim": False,
        "reward_norm": True,
        "rs_ext": 1.25,
        "rd_ext": 0.4,
        "ir_drop_n_iter": 3,
        "ir_drop_maxfev": 123,
        "nrmse_threshold": 7.5,
    }


def test_eehemt_checkpoint_config_ranks_by_lowest_nrmse():
    checkpoint_config = eehemt_ppo.build_checkpoint_config()

    assert checkpoint_config.checkpoint_score_attribute == "env_runners/min_nrmse"
    assert checkpoint_config.checkpoint_score_order == "min"


def test_hardness_env_config_uses_model_package_path_and_reward_settings(
    monkeypatch,
):
    monkeypatch.setenv("HARDNESS_MODEL_PACKAGE_PATH", "custom/hardness.zip")
    monkeypatch.setenv("HARDNESS_THRESHOLD", "725.0")
    monkeypatch.setenv("HARDNESS_REWARD_SCALE", "125.0")
    monkeypatch.setenv("HARDNESS_REWARD_MIN", "-4.0")
    monkeypatch.setenv("HARDNESS_REWARD_MAX", "5.0")
    parser = build_common_arg_parser("/project")
    hardness_ppo.add_env_args(parser, "/project")

    args = parser.parse_args([])
    env_config = hardness_ppo.build_env_config(args)

    assert env_config == {
        "model_package_path": "/project/custom/hardness.zip",
        "hardness_threshold": 725.0,
        "reward_scale": 125.0,
        "reward_min": -4.0,
        "reward_max": 5.0,
    }


def test_hardness_checkpoint_config_ranks_by_highest_predicted_hardness():
    checkpoint_config = hardness_ppo.build_checkpoint_config()

    assert (
        checkpoint_config.checkpoint_score_attribute
        == "env_runners/max_predicted_hardness"
    )
    assert checkpoint_config.checkpoint_score_order == "max"


def test_train_ppo_defaults_to_hardness_env():
    args = build_arg_parser("/project").parse_args([])

    assert args.env == "hardness"


def test_train_ppo_script_calls_new_entrypoint():
    project_root = Path(__file__).resolve().parents[1]
    script = (project_root / "scripts" / "train_ppo.sh").read_text()

    assert "python train_ppo.py" in script
    assert "train_ppo_tune.py" not in script


def test_train_ppo_can_build_eehemt_parser_from_argv():
    parser = build_arg_parser("/project", ["--env", "eehemt"])

    args = parser.parse_args(
        ["--env", "eehemt", "--va_file_path", "/tmp/model.va"]
    )

    assert args.env == "eehemt"
    assert args.va_file_path == "/tmp/model.va"


def test_select_training_module_dispatches_by_env():
    hardness_module = select_training_module("hardness")
    eehemt_module = select_training_module("eehemt")

    assert (
        hardness_module.HARDNESS_WANDB_PROJECT
        == "PPO_for_material_hardness_optimization"
    )
    assert (
        eehemt_module.EEHEMT_WANDB_PROJECT
        == "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"
    )


def test_eehemt_ppo_config_wires_callbacks_and_driver_evaluation():
    config = eehemt_ppo.build_ppo_config(
        _eehemt_args(lr=2e-5),
        num_learners=3,
        num_gpus_per_learner=0.5,
    )

    assert config.env is EEHEMTEnv_Measure_VDS
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
    assert config.num_epochs == 2
    assert config.minibatch_size == 64
    assert config.lr == pytest.approx(6e-5)
    assert config.entropy_coeff == 0.01
    assert config.grad_clip == 1.0
    assert config.vf_loss_coeff == 0.1
    assert config.vf_clip_param == 20.0
    assert config.num_learners == 3
    assert config.num_gpus_per_learner == 0.5


def test_hardness_ppo_config_wires_callbacks_and_hardness_evaluation():
    config = hardness_ppo.build_ppo_config(
        _hardness_args(lr=3e-5, evaluation_interval=4, evaluation_num_env_runners=2),
        num_learners=2,
        num_gpus_per_learner=0.0,
    )

    assert config.env is MaterialHardnessEnv
    assert config.callbacks_class is HardnessMetricsCallback
    assert config.custom_evaluation_function is evaluate_and_save_hardness
    assert config.evaluation_interval == 4
    assert config.evaluation_num_env_runners == 2
    assert config.evaluation_duration == 1
    assert config.evaluation_duration_unit == "episodes"
    assert config.evaluation_config["explore"] is False
    assert config.num_env_runners == 2
    assert config.observation_filter == "NoFilter"
    assert config.train_batch_size_per_learner == 128
    assert config.num_epochs == 2
    assert config.minibatch_size == 64
    assert config.lr == pytest.approx(6e-5)
    assert config.entropy_coeff == 0.01
    assert config.grad_clip == 1.0
    assert config.vf_loss_coeff == 0.1
    assert config.vf_clip_param == 20.0
    assert config.num_learners == 2
    assert config.num_gpus_per_learner == 0.0


def test_wandb_project_constants_are_environment_specific():
    assert (
        hardness_ppo.HARDNESS_WANDB_PROJECT
        == "PPO_for_material_hardness_optimization"
    )
    assert (
        eehemt_ppo.EEHEMT_WANDB_PROJECT
        == "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"
    )


def test_build_arg_parser_rejects_unknown_env():
    with pytest.raises(SystemExit):
        build_arg_parser("/project", ["--env", "nosuch"])
