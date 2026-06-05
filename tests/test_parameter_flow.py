import numpy as np
import pandas as pd

import env.parameter_flow as parameter_flow
from env.parameter_flow import (
    ArcsinhHuberMetric,
    EEHEMTSimulator,
    MeasuredCurveDataset,
    ParameterSpecCollection,
)


def test_parameter_specs_clamp_initial_values_and_apply_derived_constraints():
    specs = ParameterSpecCollection.from_config(
        {
            "Gamma": {"min": 0.05, "max": 0.3, "factor": 0.004},
            "Vco": {"min": -0.6, "max": -0.3, "factor": 0.006},
            "DVcoVgo": {"min": 0.001, "max": 0.15, "factor": 0.003},
            "Vtso": {"min": -0.8, "max": -0.3, "factor": 0.01},
            "DVtsoVto": {"min": 0.1, "max": 0.6, "factor": 0.006},
        },
        ["Gamma", "Vco", "DVcoVgo", "Vtso", "DVtsoVto"],
    )

    params = {
        "Gamma": 0.04,
        "Vco": -0.3,
        "DVcoVgo": 0.01,
        "Vtso": -0.5,
        "DVtsoVto": 0.2,
    }

    normalized = specs.normalize_params(params)

    assert normalized["Gamma"] == 0.05
    assert normalized["Vgo"] == -0.31
    assert normalized["Vto"] == -0.7


def test_measured_curve_dataset_selects_default_vds_window(tmp_path):
    csv_path = tmp_path / "curves.csv"
    df = pd.DataFrame(
        {
            "vg": [-1.0, 0.0],
            "0.0": [0.0, 0.0],
            "0.1": [1e-9, 2e-9],
            "0.2": [3e-9, 4e-9],
            "1.5": [5e-9, 6e-9],
        }
    )
    df.to_csv(csv_path, index=False)

    dataset = MeasuredCurveDataset.from_csv(csv_path, default_extra_vds=[1.5])

    assert dataset.vds == [0.1, 0.2, 1.5]
    assert dataset.current_matrix.shape == (3, 2)
    assert np.allclose(dataset.current_by_vds[1.5], np.array([5e-9, 6e-9]))


def test_arcsinh_huber_metric_reports_loss_and_scaled_reward_consistently():
    metric = ArcsinhHuberMetric(delta=1.0, epsilon=1e-15)
    measured = np.array([[0.0, 1e-6, 2e-6]])
    simulated = np.array([[0.0, 1.5e-6, 1e-6]])

    loss = metric.loss(measured, simulated)
    reward = metric.scaled_reward_from_loss(loss, reward_min=-5.0, reward_max=5.0)

    assert loss >= 0.0
    assert reward == np.clip(-np.log10(loss + metric.epsilon), -5.0, 5.0)
    assert metric.is_success(measured, measured, threshold=1e-12)


class _FakeCurrentFunction:
    def eval(self, *, temperature, voltages, **params):
        return np.asarray(voltages["br_disi"], dtype=float) * params["gain"]


class _FakeModel:
    def __init__(self) -> None:
        self.functions = {"I_total": _FakeCurrentFunction()}


def test_ir_drop_solver_records_non_convergence(monkeypatch):
    def fake_fsolve(*, func, x0, maxfev, full_output):
        return np.asarray(x0, dtype=float), {}, 2, "iteration is not making progress"

    monkeypatch.setattr(parameter_flow, "fsolve", fake_fsolve)
    simulator = EEHEMTSimulator(
        _FakeModel(),
        temperature=300,
        rs_ext=0.0,
        rd_ext=0.4,
        ir_drop_n_iter=1,
        ir_drop_maxfev=5,
    )

    simulator.simulate_current_matrix(
        params={"Rs": 1.0, "Rd": 2.0, "gain": 0.1},
        vgs=np.array([0.0, 1.0]),
        vds_values=[0.1, 0.2],
    )

    assert [
        {
            "vds": diagnostic["vds"],
            "converged": diagnostic["converged"],
            "ier": diagnostic["ier"],
            "message": diagnostic["message"],
        }
        for diagnostic in simulator.last_solver_diagnostics
    ] == [
        {
            "vds": 0.1,
            "converged": False,
            "ier": 2,
            "message": "iteration is not making progress",
        },
        {
            "vds": 0.2,
            "converged": False,
            "ier": 2,
            "message": "iteration is not making progress",
        },
    ]
    assert all(
        "residual_max_abs" in diagnostic
        for diagnostic in simulator.last_solver_diagnostics
    )


def test_ir_drop_solver_falls_back_when_warmup_start_misses_root(monkeypatch):
    calls = []

    def fake_fsolve(*, func, x0, maxfev, full_output):
        x0_array = np.asarray(x0, dtype=float)
        calls.append(x0_array.copy())
        if np.allclose(x0_array, np.zeros_like(x0_array)):
            return np.full_like(x0_array, 0.123), {}, 1, "solution converged"
        return x0_array, {}, 5, "iteration is not making progress"

    monkeypatch.setattr(parameter_flow, "fsolve", fake_fsolve)
    simulator = EEHEMTSimulator(
        _FakeModel(),
        temperature=300,
        rs_ext=0.0,
        rd_ext=0.4,
        ir_drop_n_iter=1,
        ir_drop_maxfev=5,
    )

    result = simulator.simulate_current_matrix(
        params={"Rs": 1.0, "Rd": 2.0, "gain": 0.1},
        vgs=np.array([0.0, 1.0]),
        vds_values=[0.1],
    )

    assert np.allclose(result[0], np.array([0.123, 0.123]))
    assert len(calls) == 1
    assert np.allclose(calls[0], np.zeros(2))
    assert simulator.last_solver_diagnostics[0]["converged"] is True
    assert simulator.last_solver_diagnostics[0]["selected_start"] == "zero"
    assert simulator.last_solver_diagnostics[0]["attempts"][0]["accepted"] is True


def test_ir_drop_solver_uses_previous_vds_solution_as_continuation(monkeypatch):
    calls = []

    def fake_fsolve(*, func, x0, maxfev, full_output):
        x0_array = np.asarray(x0, dtype=float)
        calls.append(x0_array.copy())
        if len(calls) == 1:
            return np.full_like(x0_array, 0.2), {}, 1, "first converged"
        if np.allclose(x0_array, np.full_like(x0_array, 0.2)):
            return np.full_like(x0_array, 0.3), {}, 1, "continuation converged"
        return x0_array, {}, 5, "iteration is not making progress"

    monkeypatch.setattr(parameter_flow, "fsolve", fake_fsolve)
    simulator = EEHEMTSimulator(
        _FakeModel(),
        temperature=300,
        rs_ext=0.0,
        rd_ext=0.4,
        ir_drop_n_iter=1,
        ir_drop_maxfev=5,
    )

    result = simulator.simulate_current_matrix(
        params={"Rs": 1.0, "Rd": 2.0, "gain": 0.1},
        vgs=np.array([0.0, 1.0]),
        vds_values=[0.1, 0.2],
    )

    assert np.allclose(result[0], np.array([0.2, 0.2]))
    assert np.allclose(result[1], np.array([0.3, 0.3]))
    assert np.allclose(calls[0], np.zeros(2))
    assert np.allclose(calls[1], np.array([0.2, 0.2]))
    assert simulator.last_solver_diagnostics[1]["selected_start"] == "continuation"
