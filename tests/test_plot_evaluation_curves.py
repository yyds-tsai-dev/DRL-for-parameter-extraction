from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from utils.plot import _format_loss_for_filename, save_evaluation_iv_curves


def test_save_evaluation_iv_curves_writes_historical_and_latest_files(tmp_path):
    curve_condition_values = [0.1, 0.2]
    plot_data = {
        "vgs": np.array([0.0, 1.0]),
        "i_meas_dict": {
            0.1: np.array([1.0e-3, 2.0e-3]),
            0.2: np.array([2.0e-3, 3.0e-3]),
        },
        "i_sim_current_matrix": np.array(
            [
                [1.1e-3, 2.1e-3],
                [2.1e-3, 3.1e-3],
            ]
        ),
    }
    loss = 0.001234

    paths = save_evaluation_iv_curves(
        curve_condition_values=curve_condition_values,
        plot_data=plot_data,
        plot_dir=str(tmp_path),
        evaluation_index=7,
        training_iteration=42,
        fit_loss=loss,
    )

    assert set(paths) == {"linear", "log", "latest_linear", "latest_log"}
    for path in paths.values():
        assert Path(path).exists()

    expected_stem = "eval_000007_iter_000042_loss_1.234e-03"
    assert expected_stem in Path(paths["linear"]).name
    assert expected_stem in Path(paths["log"]).name
    assert Path(paths["linear"]).name == f"{expected_stem}.png"
    assert Path(paths["log"]).name == f"{expected_stem}_log.png"
    assert Path(paths["latest_linear"]).name == "latest_eval.png"
    assert Path(paths["latest_log"]).name == "latest_eval_log.png"


def test_save_evaluation_iv_curves_preserves_history_when_latest_is_overwritten(
    tmp_path,
):
    curve_condition_values = [0.1, 0.2]
    first_plot_data = {
        "vgs": np.array([0.0, 1.0]),
        "i_meas_dict": {
            0.1: np.array([1.0e-3, 2.0e-3]),
            0.2: np.array([2.0e-3, 3.0e-3]),
        },
        "i_sim_current_matrix": np.array(
            [
                [1.1e-3, 2.1e-3],
                [2.1e-3, 3.1e-3],
            ]
        ),
    }
    second_plot_data = {
        "vgs": np.array([0.0, 1.0]),
        "i_meas_dict": {
            0.1: np.array([8.0e-3, 1.2e-2]),
            0.2: np.array([1.6e-2, 2.4e-2]),
        },
        "i_sim_current_matrix": np.array(
            [
                [9.0e-3, 1.3e-2],
                [1.7e-2, 2.5e-2],
            ]
        ),
    }

    first_paths = save_evaluation_iv_curves(
        curve_condition_values=curve_condition_values,
        plot_data=first_plot_data,
        plot_dir=str(tmp_path),
        evaluation_index=1,
        training_iteration=10,
        fit_loss=0.001,
    )
    second_paths = save_evaluation_iv_curves(
        curve_condition_values=curve_condition_values,
        plot_data=second_plot_data,
        plot_dir=str(tmp_path),
        evaluation_index=2,
        training_iteration=20,
        fit_loss=0.002,
    )

    for key in ("linear", "log"):
        assert Path(first_paths[key]).exists()
        assert Path(second_paths[key]).exists()

    assert Path(second_paths["latest_linear"]).read_bytes() == Path(
        second_paths["linear"]
    ).read_bytes()
    assert Path(second_paths["latest_log"]).read_bytes() == Path(
        second_paths["log"]
    ).read_bytes()
    assert Path(second_paths["latest_linear"]).read_bytes() != Path(
        first_paths["linear"]
    ).read_bytes()
    assert Path(second_paths["latest_log"]).read_bytes() != Path(
        first_paths["log"]
    ).read_bytes()


def test_save_evaluation_iv_curves_adds_suffix_when_historical_files_collide(
    tmp_path,
):
    curve_condition_values = [0.1, 0.2]
    first_plot_data = {
        "vgs": np.array([0.0, 1.0]),
        "i_meas_dict": {
            0.1: np.array([1.0e-3, 2.0e-3]),
            0.2: np.array([2.0e-3, 3.0e-3]),
        },
        "i_sim_current_matrix": np.array(
            [
                [1.1e-3, 2.1e-3],
                [2.1e-3, 3.1e-3],
            ]
        ),
    }
    second_plot_data = {
        "vgs": np.array([0.0, 1.0]),
        "i_meas_dict": {
            0.1: np.array([6.0e-3, 8.0e-3]),
            0.2: np.array([1.2e-2, 1.8e-2]),
        },
        "i_sim_current_matrix": np.array(
            [
                [7.0e-3, 9.0e-3],
                [1.3e-2, 1.9e-2],
            ]
        ),
    }
    save_kwargs = {
        "curve_condition_values": curve_condition_values,
        "plot_dir": str(tmp_path),
        "evaluation_index": 3,
        "training_iteration": 30,
        "fit_loss": 0.003,
    }

    first_paths = save_evaluation_iv_curves(
        plot_data=first_plot_data,
        **save_kwargs,
    )
    second_paths = save_evaluation_iv_curves(
        plot_data=second_plot_data,
        **save_kwargs,
    )

    assert Path(first_paths["linear"]).exists()
    assert Path(first_paths["log"]).exists()
    assert second_paths["linear"] != first_paths["linear"]
    assert second_paths["log"] != first_paths["log"]
    assert "_dup001" in Path(second_paths["linear"]).name
    assert "_dup001_log" in Path(second_paths["log"]).name
    assert Path(second_paths["latest_linear"]).read_bytes() == Path(
        second_paths["linear"]
    ).read_bytes()
    assert Path(second_paths["latest_log"]).read_bytes() == Path(
        second_paths["log"]
    ).read_bytes()
    assert Path(second_paths["latest_linear"]).read_bytes() != Path(
        first_paths["linear"]
    ).read_bytes()
    assert Path(second_paths["latest_log"]).read_bytes() != Path(
        first_paths["log"]
    ).read_bytes()


def test_format_loss_for_filename_handles_unknown_loss():
    assert _format_loss_for_filename(None) == "unknown"
