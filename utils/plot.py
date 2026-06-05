import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _format_loss_for_filename(loss: float | None) -> str:
    if loss is None:
        return "unknown"
    return f"{loss:.3e}"


def _curve_condition_names() -> list[str]:
    names = os.getenv("CURVE_CONDITION_NAMES", "UGW,NOF")
    parsed_names = [name.strip() for name in names.split(",") if name.strip()]
    return parsed_names or ["UGW", "NOF"]


def _historical_eval_paths(plot_dir: str, stem: str) -> tuple[str, str]:
    suffix = ""
    duplicate_index = 0
    while True:
        candidate_stem = f"{stem}{suffix}"
        linear_path = os.path.join(plot_dir, f"{candidate_stem}.png")
        log_path = os.path.join(plot_dir, f"{candidate_stem}_log.png")
        if not os.path.exists(linear_path) and not os.path.exists(log_path):
            return linear_path, log_path

        duplicate_index += 1
        suffix = f"_dup{duplicate_index:03d}"


def plot_all_condition_iv_curve(
    curve_condition_values: list,
    plot_data: dict,
    plot_dir: str,
    log_y: bool = True,
    plot_cnt: int = 0,
    save_path: str | None = None,
):
    """
    Plots and saves I-V curves for all (Ugw, NOF) conditions on a single graph.
    Each curve type (Target, Initial, Current) has its own color gradient.

    Args:
        curve_condition_values (list): A list containing all (Ugw, NOF) float values.
        plot_data (dict): A dictionary containing static plotting data.
        plot_dir (str): The directory path to save the plots.
    """
    # print(f"i_sim_current_matrix shape: {plot_data['i_sim_current_matrix'].shape}")
    # === Get static data from plot_data ===
    vgs = plot_data["vgs"]
    i_meas_dict = plot_data["i_meas_dict"]
    # i_sim_init_matrix = plot_data["i_sim_init_matrix"]
    i_sim_current_matrix = plot_data["i_sim_current_matrix"]
    curve_condition_names = _curve_condition_names()

    fig, ax = plt.subplots(figsize=(10, 7))

    # === Create distinct color maps for each curve type ===
    # We generate a list of colors for each type of curve.
    # Using np.linspace(0.5, 1, ...) ensures colors are not too light.
    num_curves = len(curve_condition_values)
    target_colors = plt.get_cmap("rainbow")(np.linspace(0.5, 1, num_curves))
    current_colors = plt.get_cmap("rainbow")(np.linspace(0.5, 1, num_curves))

    vds_legend_handles = []
    vds_legend_labels = []

    # === Iterate through each (Ugw, NOF) pair and plot with gradient colors ===
    for i, condition_value in enumerate(curve_condition_values):
        # label_target = "Experiments" if i == len(curve_condition_values) - 1 else None
        # label_current = "Modeling" if i == len(curve_condition_values) - 1 else None
        
        # 1. Plot the target data (Measured) using the 'Blues' colormap.
        ax.plot(
            vgs,
            i_meas_dict[condition_value],
            marker="o",
            linestyle="None",
            color=target_colors[i],
            # label=label_target,
            ms=3,
        )

        # 2. Plot the current simulation using the 'Reds' colormap.
        ax.plot(
            vgs,
            i_sim_current_matrix[i, :],
            linestyle="-",  # Set line style to solid
            color=current_colors[i],
            # label=label_current,
        )

        vds_legend_handles.append(Line2D([0], [0], color=current_colors[i], lw=2))
        vds_legend_labels.append(f"{condition_value}V")


    # === Set the plot style and labels ===
    ax.set_title(f"I-V Curve Comparison for All {', '.join(curve_condition_names)} Values")
    ax.set_xlabel("Vg(Gate Voltage) [V]")
    if log_y:
        ax.set_ylabel("log(Id) (Log Drain Current) [mA]")
        ax.set_yscale("log")
        if save_path is None:
            save_path = os.path.join(
                plot_dir,
                f"iv_curve_all_{'_'.join(curve_condition_names)}_log_{plot_cnt}.png",
            )
    else:
        ax.set_ylabel("Id(Drain Current) [mA]")
        if save_path is None:
            save_path = os.path.join(
                plot_dir,
                f"iv_curve_all_{'_'.join(curve_condition_names)}_{plot_cnt}.png",
            )

    ax.grid(True, which="both", ls="--", alpha=0.7)
    # ax.legend(loc="best")
    # 1. Main Legend: Experiments (Points) & Modeling (Lines)
    legend_elements_main = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label="Experiments",
            linestyle="None",
            markersize=5,
        ),
        Line2D([0], [0], color="black", label="Modeling", linestyle="-", lw=2),
    ]
    legend1 = ax.legend(handles=legend_elements_main, loc="upper left")
    ax.add_artist(legend1)

    # 2. Vds Legend (Right Side)
    legend_title = curve_condition_names[0] if curve_condition_names else "Vds"
    ax.legend(
        handles=vds_legend_handles,
        labels=vds_legend_labels,
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    # === Save the plot ===
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info("==== I-V curves plot saved in %s ====", save_path)


def save_evaluation_iv_curves(
    *,
    curve_condition_values: list,
    plot_data: dict,
    plot_dir: str,
    evaluation_index: int,
    training_iteration: int,
    fit_loss: float | None,
) -> dict[str, str]:
    os.makedirs(plot_dir, exist_ok=True)
    loss_label = _format_loss_for_filename(fit_loss)
    stem = (
        f"eval_{evaluation_index:06d}_"
        f"iter_{training_iteration:06d}_"
        f"loss_{loss_label}"
    )
    linear_path, log_path = _historical_eval_paths(plot_dir, stem)
    latest_linear_path = os.path.join(plot_dir, "latest_eval.png")
    latest_log_path = os.path.join(plot_dir, "latest_eval_log.png")

    plot_all_condition_iv_curve(
        curve_condition_values=curve_condition_values,
        plot_data=plot_data,
        plot_dir=plot_dir,
        log_y=False,
        save_path=linear_path,
    )
    plot_all_condition_iv_curve(
        curve_condition_values=curve_condition_values,
        plot_data=plot_data,
        plot_dir=plot_dir,
        log_y=True,
        save_path=log_path,
    )
    shutil.copyfile(linear_path, latest_linear_path)
    shutil.copyfile(log_path, latest_log_path)

    return {
        "linear": linear_path,
        "log": log_path,
        "latest_linear": latest_linear_path,
        "latest_log": latest_log_path,
    }
