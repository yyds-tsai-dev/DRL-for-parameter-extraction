import numpy as np
import pandas as pd

from env.parameter_flow import (
    ArcsinhHuberMetric,
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
