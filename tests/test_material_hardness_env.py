from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.material_hardness_env import MaterialHardnessEnv


class FakeInferenceModel:
    def __init__(self, model_package_path):
        self.model_package_path = model_package_path
        self.inputs = []

    def predict(self, input_data, include_input=True):
        df = pd.DataFrame(input_data)
        self.inputs.append(df.copy())
        hardness = 650.0 + 100.0 * float(df.iloc[0]["frac_Ni"])
        return pd.DataFrame(
            {
                "Predicted hardness": [hardness],
                "Uncertainty hardness": [12.5],
            }
        )


def make_env():
    return MaterialHardnessEnv(
        {
            "model_package_path": "/tmp/fake.zip",
            "inference_model_cls": FakeInferenceModel,
            "hardness_threshold": 650.0,
            "reward_scale": 100.0,
            "reward_min": -3.0,
            "reward_max": 3.0,
        }
    )


def fixed_prediction_model_cls(predicted_hardness, uncertainty_hardness=12.5):
    class FixedPredictionInferenceModel(FakeInferenceModel):
        def predict(self, input_data, include_input=True):
            df = pd.DataFrame(input_data)
            self.inputs.append(df.copy())
            result = {"Predicted hardness": [predicted_hardness]}
            if uncertainty_hardness is not None:
                result["Uncertainty hardness"] = [uncertainty_hardness]
            return pd.DataFrame(result)

    return FixedPredictionInferenceModel


def test_reset_returns_fixed_observation_inside_space():
    env = make_env()

    observation, info = env.reset(seed=123)

    assert observation.shape == (1,)
    assert env.observation_space.contains(observation)
    assert info == {}


def test_step_projects_action_injects_fixed_features_and_terminates():
    env = make_env()
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(
        np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    )

    composition = info["composition"]
    tunable_sum = sum(
        composition[name]
        for name in ("frac_Al", "frac_Cr", "frac_Mn", "frac_Fe", "frac_Co", "frac_Ni")
    )
    assert env.observation_space.contains(observation)
    assert terminated is True
    assert truncated is False
    assert np.isclose(tunable_sum, 1.0)
    assert all(
        0.05 <= composition[name] <= 0.35
        for name in composition
        if name not in {"frac_Cu", "frac_Mo"}
    )
    assert composition["frac_Cu"] == 0.0
    assert composition["frac_Mo"] == 0.0
    assert info["predicted_hardness"] >= 650.0
    assert info["uncertainty_hardness"] == 12.5
    assert info["is_success"] is True


@pytest.mark.parametrize(
    ("predicted_hardness", "expected_reward", "expected_unclipped", "expected_success"),
    [
        (550.0, -1.0, -1.0, False),
        (650.0, 0.0, 0.0, True),
        (1000.0, 3.0, 3.5, True),
    ],
)
def test_step_computes_reward_from_predicted_hardness(
    predicted_hardness,
    expected_reward,
    expected_unclipped,
    expected_success,
):
    env = MaterialHardnessEnv(
        {
            "model_package_path": "/tmp/fake.zip",
            "inference_model_cls": fixed_prediction_model_cls(predicted_hardness),
            "hardness_threshold": 650.0,
            "reward_scale": 100.0,
            "reward_min": -3.0,
            "reward_max": 3.0,
        }
    )
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert reward == expected_reward
    assert info["reward_unclipped"] == expected_unclipped
    assert info["predicted_hardness"] == predicted_hardness
    assert info["is_success"] is expected_success
    assert terminated is True
    assert truncated is False


@pytest.mark.parametrize("predicted_hardness", [np.nan, np.inf])
@pytest.mark.parametrize("uncertainty_hardness", [None, np.inf])
def test_step_handles_non_finite_prediction_as_failed_episode(
    predicted_hardness,
    uncertainty_hardness,
):
    env = MaterialHardnessEnv(
        {
            "model_package_path": "/tmp/fake.zip",
            "inference_model_cls": fixed_prediction_model_cls(
                predicted_hardness,
                uncertainty_hardness=uncertainty_hardness,
            ),
        }
    )
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert reward == -3.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is False
    assert np.isfinite(info["predicted_hardness"])
    assert info["predicted_hardness"] == 350.0
    assert np.isfinite(info["reward_unclipped"])
    assert info["reward_unclipped"] == -3.0
    assert np.isfinite(info["uncertainty_hardness"])
    assert info["uncertainty_hardness"] == 0.0
    assert info["raw_predicted_hardness"] == str(float(predicted_hardness))
    assert "non-finite" in info["error"]


@pytest.mark.parametrize(
    "action",
    [
        np.zeros(5, dtype=np.float32),
        np.zeros((1, 6), dtype=np.float32),
    ],
)
def test_step_rejects_action_not_shape_six(action):
    env = make_env()
    env.reset(seed=123)

    with pytest.raises(ValueError, match=r"action shape must be \(6,\)"):
        env.step(action)


def test_close_forwards_to_model_when_available():
    class CloseableInferenceModel(FakeInferenceModel):
        def __init__(self, model_package_path):
            super().__init__(model_package_path)
            self.closed = False

        def close(self):
            self.closed = True

    env = MaterialHardnessEnv(
        {
            "model_package_path": "/tmp/fake.zip",
            "inference_model_cls": CloseableInferenceModel,
        }
    )

    env.close()

    assert env.model.closed is True


def test_init_rejects_invalid_reward_config():
    with pytest.raises(ValueError, match="reward_scale"):
        MaterialHardnessEnv(
            {
                "model_package_path": "/tmp/fake.zip",
                "inference_model_cls": FakeInferenceModel,
                "reward_scale": 0.0,
            }
        )

    with pytest.raises(ValueError, match="reward_min"):
        MaterialHardnessEnv(
            {
                "model_package_path": "/tmp/fake.zip",
                "inference_model_cls": FakeInferenceModel,
                "reward_min": 4.0,
                "reward_max": 3.0,
            }
        )
