"""Registration glue for the material-hardness optimization problem."""

from problems.registry import ProblemSpec
from training import hardness_ppo


def build_spec() -> ProblemSpec:
    return ProblemSpec(
        name="hardness",
        module=hardness_ppo,
        wandb_project=hardness_ppo.HARDNESS_WANDB_PROJECT,
        checkpoint_metric="env_runners/max_predicted_hardness",
        checkpoint_order="max",
        add_env_args=hardness_ppo.add_env_args,
        build_env_config=hardness_ppo.build_env_config,
        build_ppo_config=hardness_ppo.build_ppo_config,
        build_checkpoint_config=hardness_ppo.build_checkpoint_config,
    )
