"""Registration glue for the EEHEMT parameter-extraction problem."""

from env.objectives import NRMSEMinimizeObjective
from problems.registry import ProblemSpec
from training import eehemt_ppo


def build_spec() -> ProblemSpec:
    return ProblemSpec(
        name="eehemt",
        module=eehemt_ppo,
        wandb_project=eehemt_ppo.EEHEMT_WANDB_PROJECT,
        checkpoint_metric=NRMSEMinimizeObjective.RANKED_METRIC,
        checkpoint_order=NRMSEMinimizeObjective.RANKED_ORDER,
        add_env_args=eehemt_ppo.add_env_args,
        build_env_config=eehemt_ppo.build_env_config,
        build_ppo_config=eehemt_ppo.build_ppo_config,
        build_checkpoint_config=eehemt_ppo.build_checkpoint_config,
    )
