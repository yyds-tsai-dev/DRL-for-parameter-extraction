import os

import numpy as np

from env.eehemt_env import EEHEMTEnv_Measure_VDS


def _env_config() -> dict[str, object]:
    cwd = os.getcwd()
    return {
        "va_file_path": os.path.join(cwd, os.getenv("VA_FILE_PATH", "")),
        "csv_file_path": os.path.join(cwd, os.getenv("CSV_FILE_PATH", "")),
        "random_init": False,
        "reduce_obs_err_dim": False,
        "reward_norm": False,
    }


def test_reset_observation_is_inside_declared_space():
    env = EEHEMTEnv_Measure_VDS(_env_config())

    observation, info = env.reset(seed=123)

    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert "arcsinh_huber_loss" in info


def test_random_reduced_reset_observation_is_inside_declared_space():
    config = _env_config()
    config["random_init"] = True
    config["reduce_obs_err_dim"] = True
    env = EEHEMTEnv_Measure_VDS(config)

    observation, info = env.reset(seed=123)

    assert observation.dtype == np.float32
    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    assert "arcsinh_huber_loss" in info


def test_termination_uses_arcsinh_huber_threshold(monkeypatch):
    monkeypatch.setenv("ARCSINH_HUBER_THRESHOLD", "999.0")
    monkeypatch.setenv("REWARD_MIN", "-5.0")
    monkeypatch.setenv("REWARD_MAX", "5.0")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert terminated is True
    assert truncated is False
    assert reward == np.clip(
        -np.log10(info["arcsinh_huber_loss"] + 1e-15),
        -5.0,
        5.0,
    )
