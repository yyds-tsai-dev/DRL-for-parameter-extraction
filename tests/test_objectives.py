from __future__ import annotations

import numpy as np
import pytest

from env.objectives import NRMSEMinimizeObjective, ThresholdMaximizeObjective


def make_threshold_objective(**overrides):
    kwargs = {
        "threshold": 650.0,
        "scale": 100.0,
        "reward_min": -3.0,
        "reward_max": 3.0,
    }
    kwargs.update(overrides)
    return ThresholdMaximizeObjective(**kwargs)


@pytest.mark.parametrize(
    ("value", "expected_reward", "expected_unclipped", "expected_success"),
    [
        (550.0, -1.0, -1.0, False),
        (650.0, 0.0, 0.0, True),
        (1000.0, 3.0, 3.5, True),
        (250.0, -3.0, -4.0, False),
    ],
)
def test_threshold_objective_reproduces_hardness_reward_table(
    value, expected_reward, expected_unclipped, expected_success
):
    outcome = make_threshold_objective().evaluate(value)

    assert outcome.reward == expected_reward
    assert outcome.reward_unclipped == expected_unclipped
    assert outcome.success is expected_success


def test_threshold_objective_success_is_inclusive_at_threshold():
    outcome = make_threshold_objective().evaluate(650.0)

    assert outcome.success is True


def test_threshold_objective_validates_scale_and_bounds():
    with pytest.raises(ValueError, match="reward_scale must be positive"):
        make_threshold_objective(scale=0.0)

    with pytest.raises(
        ValueError, match="reward_min must be less than or equal to reward_max"
    ):
        make_threshold_objective(reward_min=4.0, reward_max=3.0)


def test_threshold_objective_ranking_identity():
    assert (
        ThresholdMaximizeObjective.RANKED_METRIC
        == "env_runners/max_predicted_hardness"
    )
    assert ThresholdMaximizeObjective.RANKED_ORDER == "max"


def make_nrmse_objective(**overrides):
    kwargs = {
        "threshold": 10.0,
        "reward_min": -5.0,
        "reward_max": 5.0,
        "epsilon": 1e-15,
    }
    kwargs.update(overrides)
    return NRMSEMinimizeObjective(**kwargs)


@pytest.mark.parametrize("nrmse", [5.0, 0.05, 37.5])
def test_nrmse_objective_matches_reference_formula(nrmse):
    objective = make_nrmse_objective()

    reward = objective.reward_from_nrmse(nrmse)

    assert reward == float(
        np.clip(-np.log10((nrmse / 100.0) + 1e-15), -5.0, 5.0)
    )


def test_nrmse_objective_clips_at_both_bounds():
    objective = make_nrmse_objective()

    assert objective.reward_from_nrmse(0.0) == 5.0
    assert objective.reward_from_nrmse(1e9) == -5.0


def test_nrmse_objective_success_is_strictly_below_threshold():
    objective = make_nrmse_objective()

    assert objective.is_success(9.999) is True
    assert objective.is_success(10.0) is False


def test_nrmse_objective_ranking_identity():
    assert NRMSEMinimizeObjective.RANKED_METRIC == "env_runners/min_nrmse"
    assert NRMSEMinimizeObjective.RANKED_ORDER == "min"
