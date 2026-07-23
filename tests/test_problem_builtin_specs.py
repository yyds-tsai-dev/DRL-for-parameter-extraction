import problems  # noqa: F401  (import triggers self-registration)
from problems import registry
from training import eehemt_ppo, hardness_ppo


def test_builtin_problem_names():
    assert registry.names() == ["eehemt", "hardness"]


def test_hardness_spec_points_at_existing_module_parts():
    spec = registry.get("hardness")

    assert spec.module is hardness_ppo
    assert spec.wandb_project == "PPO_for_material_hardness_optimization"
    assert spec.checkpoint_metric == "env_runners/max_predicted_hardness"
    assert spec.checkpoint_order == "max"
    assert spec.add_env_args is hardness_ppo.add_env_args
    assert spec.build_env_config is hardness_ppo.build_env_config
    assert spec.build_ppo_config is hardness_ppo.build_ppo_config
    assert spec.build_checkpoint_config is hardness_ppo.build_checkpoint_config


def test_eehemt_spec_points_at_existing_module_parts():
    spec = registry.get("eehemt")

    assert spec.module is eehemt_ppo
    assert spec.wandb_project == "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"
    assert spec.checkpoint_metric == "env_runners/min_nrmse"
    assert spec.checkpoint_order == "min"
    assert spec.add_env_args is eehemt_ppo.add_env_args
    assert spec.build_env_config is eehemt_ppo.build_env_config
    assert spec.build_ppo_config is eehemt_ppo.build_ppo_config
    assert spec.build_checkpoint_config is eehemt_ppo.build_checkpoint_config
