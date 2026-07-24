from __future__ import annotations

import os

import numpy as np

import env.eehemt_env as eehemt_env_module
from env.eehemt_env import EEHEMTEnv_Measure_VDS
from env.parameter_flow import MeasuredCurveDataset


class FakeSimulator:
    def __init__(self, dataset, defaults):
        self._dataset = dataset
        self._defaults = defaults
        self.last_solver_diagnostics = []

    def modelcard_defaults(self):
        return dict(self._defaults)

    def simulate_current_matrix(self, *, params, vgs, vds_values, current_step):
        self.last_solver_diagnostics = [
            {"converged": True} for _ in range(len(vds_values))
        ]
        return np.zeros_like(self._dataset.current_matrix)


def test_simulator_factory_injects_fake_and_env_still_resets():
    csv_path = os.path.join(os.getcwd(), os.getenv("CSV_FILE_PATH", ""))
    dataset = MeasuredCurveDataset.from_csv(csv_path)
    defaults = {
        name: 0.0 for name in eehemt_env_module.PARAMETER_SPECS.names
    }
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeSimulator(dataset, defaults)

    env = EEHEMTEnv_Measure_VDS(
        {
            "va_file_path": "/nonexistent/never-compiled.va",
            "csv_file_path": csv_path,
            "random_init": False,
            "reduce_obs_err_dim": False,
            "reward_norm": False,
            "simulator_factory": factory,
        }
    )

    observation, info = env.reset(seed=123)

    assert isinstance(env.simulator, FakeSimulator)
    assert captured["va_file_path"] == "/nonexistent/never-compiled.va"
    assert set(captured) == {
        "va_file_path",
        "temperature",
        "rs_ext",
        "rd_ext",
        "ir_drop_n_iter",
        "ir_drop_maxfev",
    }
    assert env.observation_space.contains(observation)
    assert "nrmse" in info
