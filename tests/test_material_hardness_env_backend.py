from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from env.backends import PredictionResult
from env.material_hardness_env import MaterialHardnessEnv


class FakeBackend:
    predicted = 720.0
    uncertainty = 12.5

    def __init__(self, model_package_path):
        self.model_package_path = model_package_path
        self.features_seen = []
        self.closed = False

    def predict(self, features):
        self.features_seen.append(dict(features))
        return PredictionResult(
            values={"hardness": self.predicted},
            uncertainties={"hardness": self.uncertainty},
        )

    def close(self):
        self.closed = True


class LegacyFakeModel:
    def __init__(self, model_package_path):
        self.model_package_path = model_package_path

    def predict(self, input_data, include_input=True):
        return pd.DataFrame(
            {
                "Predicted hardness": [FakeBackend.predicted],
                "Uncertainty hardness": [FakeBackend.uncertainty],
            }
        )


BASE_CONFIG = {
    "model_package_path": "/tmp/fake.zip",
    "hardness_threshold": 650.0,
    "reward_scale": 100.0,
    "reward_min": -3.0,
    "reward_max": 3.0,
}


def make_backend_env(backend_cls=FakeBackend, **overrides):
    config = dict(BASE_CONFIG)
    config["prediction_backend_cls"] = backend_cls
    config.update(overrides)
    return MaterialHardnessEnv(config)


def test_backend_and_legacy_paths_produce_identical_step_results():
    backend_env = make_backend_env()
    legacy_env = MaterialHardnessEnv(
        dict(BASE_CONFIG, inference_model_cls=LegacyFakeModel)
    )
    action = np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.25], dtype=np.float32)

    backend_env.reset(seed=7)
    legacy_env.reset(seed=7)
    b_obs, b_reward, b_term, b_trunc, b_info = backend_env.step(action)
    l_obs, l_reward, l_term, l_trunc, l_info = legacy_env.step(action)

    assert b_reward == l_reward
    assert (b_term, b_trunc) == (l_term, l_trunc)
    assert b_info["predicted_hardness"] == l_info["predicted_hardness"]
    assert b_info["uncertainty_hardness"] == l_info["uncertainty_hardness"]
    assert b_info["reward_unclipped"] == l_info["reward_unclipped"]
    assert b_info["is_success"] == l_info["is_success"]
    assert b_info["composition"] == l_info["composition"]


def test_backend_receives_full_composition_including_fixed_fractions():
    env = make_backend_env()
    env.reset(seed=7)

    env.step(np.zeros(6, dtype=np.float32))

    features = env.model.features_seen[0]
    assert features["frac_Cu"] == 0.0
    assert features["frac_Mo"] == 0.0
    assert set(features) == {
        "frac_Al",
        "frac_Cr",
        "frac_Mn",
        "frac_Fe",
        "frac_Co",
        "frac_Ni",
        "frac_Cu",
        "frac_Mo",
    }


def test_backend_non_finite_prediction_follows_failure_path():
    class NanBackend(FakeBackend):
        predicted = float("nan")

    env = make_backend_env(NanBackend)
    env.reset(seed=7)

    _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert reward == -3.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is False
    assert info["predicted_hardness"] == 350.0
    assert info["uncertainty_hardness"] == 12.5
    assert "non-finite" in info["error"]


def test_backend_missing_target_raises_missing_column_keyerror():
    class WrongTargetBackend(FakeBackend):
        def predict(self, features):
            return PredictionResult(values={"density": 7.9}, uncertainties={})

    env = make_backend_env(WrongTargetBackend)
    env.reset(seed=7)

    with pytest.raises(
        KeyError, match="Prediction output missing column: Predicted hardness"
    ):
        env.step(np.zeros(6, dtype=np.float32))


def test_backend_close_forwards():
    env = make_backend_env()

    env.close()

    assert env.model.closed is True
