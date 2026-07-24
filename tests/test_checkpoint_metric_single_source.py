import problems  # noqa: F401
from env.objectives import NRMSEMinimizeObjective, ThresholdMaximizeObjective
from problems import registry
from training import eehemt_ppo, hardness_ppo


def test_hardness_metric_single_sourced_from_objective():
    spec = registry.get("hardness")
    checkpoint_config = hardness_ppo.build_checkpoint_config()

    assert spec.checkpoint_metric == ThresholdMaximizeObjective.RANKED_METRIC
    assert spec.checkpoint_order == ThresholdMaximizeObjective.RANKED_ORDER
    assert (
        checkpoint_config.checkpoint_score_attribute
        == ThresholdMaximizeObjective.RANKED_METRIC
    )
    assert (
        checkpoint_config.checkpoint_score_order
        == ThresholdMaximizeObjective.RANKED_ORDER
    )


def test_eehemt_metric_single_sourced_from_objective():
    spec = registry.get("eehemt")
    checkpoint_config = eehemt_ppo.build_checkpoint_config()

    assert spec.checkpoint_metric == NRMSEMinimizeObjective.RANKED_METRIC
    assert spec.checkpoint_order == NRMSEMinimizeObjective.RANKED_ORDER
    assert (
        checkpoint_config.checkpoint_score_attribute
        == NRMSEMinimizeObjective.RANKED_METRIC
    )
    assert (
        checkpoint_config.checkpoint_score_order
        == NRMSEMinimizeObjective.RANKED_ORDER
    )
