import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from dotenv import load_dotenv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from env.eehemt_env import CURVE_CONDITION_NAMES, key_params_names

load_dotenv()
### New
PLOT_PERIOD = int(os.getenv("PLOT_PERIOD", 5))

def plot_all_condition_iv_curve(
    curve_condition_values: list,
    plot_data: dict,
    plot_dir: str,
    log_y: bool = True,
    plot_cnt: int = 0,
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
    ax.set_title(f"I-V Curve Comparison for All {', '.join(CURVE_CONDITION_NAMES)} Values")
    ax.set_xlabel("Vg(Gate Voltage) [V]")
    if log_y:
        ax.set_ylabel("log(Id) (Log Drain Current) [mA]")
        ax.set_yscale("log")
        save_path = os.path.join(
            plot_dir, f"iv_curve_all_{'_'.join(CURVE_CONDITION_NAMES)}_log_{plot_cnt}.png"
        )
    else:
        ax.set_ylabel("Id(Drain Current) [mA]")
        save_path = os.path.join(
            plot_dir, f"iv_curve_all_{'_'.join(CURVE_CONDITION_NAMES)}_{plot_cnt}.png"
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
    legend_title = CURVE_CONDITION_NAMES[0] if CURVE_CONDITION_NAMES else "Vds"
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

    print(f"==== I-V curves plot saved in {save_path} ====")


### New
class PlotCurve(DefaultCallbacks):
    """
    RLlib Callback for plotting I-V curves at the end of each episode.
    It fetches static data (vgs, i_meas, etc.) only once and stores it.
    """

    def __init__(self):
        super().__init__()
        base_plot_dir = os.getenv("PLOT_DIR", "result/iv-curve")
        algo_name = os.getenv("ALGO_NAME", "ppo")
        today_date = datetime.now().strftime("%Y-%m-%d")
        self.plot_dir = os.path.join(base_plot_dir, algo_name, today_date)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)

        self.plot_data = None
        ### New
        self.curve_condition_values = None  # Store lg values for plotting
        self.vds = None
        self.plot_cnt = 0
        self.min_arcsinh_huber_loss = float("inf")

    def on_environment_created(
        self, *, env_runner, metrics_logger=None, env, env_context, **kwargs
    ):
        actual_env = env.envs[  # type: ignore
            0
        ].unwrapped

        if self.plot_data is None:
            print("\nFetching static plot data from the environment...\n")
            if hasattr(actual_env, "_get_plot_data_matrix"):
                # Fetch static plot data only once
                self.plot_data = actual_env._get_plot_data_matrix()
                if hasattr(actual_env, "curve_condition_values"):
                    self.curve_condition_values = actual_env.curve_condition_values
                if hasattr(actual_env, "vds"):
                    self.vds = actual_env.vds
                    if self.curve_condition_values is None:
                        self.curve_condition_values = actual_env.vds
            else:
                print("Warning: Environment does not have '_get_plot_data' method.")
                self.plot_data = {}
                return

    ### New
    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        # episode.custom_data["Vto"] = []
        tunable_params = {name: [] for name in key_params_names}
        episode.custom_data.update(tunable_params)  # type: ignore

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        current_params = env.envs[0].unwrapped.current_params  # type: ignore
        # print(f"current_params: {current_params}")
        # episode.custom_data["Vto"].append(current_params["Vto"])
        for param_name in key_params_names:
            episode.custom_data[param_name].append(current_params[param_name])  # type: ignore

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        last_info = episode.infos[-1]
        self.plot_cnt += 1
        if "i_sim_current_matrix" in last_info:
            fit_loss = last_info["arcsinh_huber_loss"]
            if fit_loss < self.min_arcsinh_huber_loss:
                self.min_arcsinh_huber_loss = fit_loss

            metrics_logger.log_value(
                "last_arcsinh_huber_loss",
                fit_loss,
                reduce="mean",
            )
            metrics_logger.log_value(
                "min_arcsinh_huber_loss",
                self.min_arcsinh_huber_loss,
                reduce="mean",
            )
            print(
                "\nFinal arcsinh Huber loss: "
                f"{fit_loss:.6g}\nMin arcsinh Huber loss: "
                f"{self.min_arcsinh_huber_loss:.6g}"
            )

            self.plot_data["i_sim_current_matrix"] = last_info["i_sim_current_matrix"]  # type: ignore

            ### New
            if self.plot_cnt % PLOT_PERIOD == 0:
                if self.curve_condition_values is not None:
                    plot_all_condition_iv_curve(
                        curve_condition_values=self.curve_condition_values,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        # log_y=os.getenv("LOG_Y", "True").lower() == "true",
                        log_y=False,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
                    plot_all_condition_iv_curve(
                        curve_condition_values=self.curve_condition_values,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        log_y=True,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
                elif self.vds is not None:
                    plot_all_condition_iv_curve(
                        curve_condition_values=self.vds,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        # log_y=os.getenv("LOG_Y", "True").lower() == "true",
                        log_y=False,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
                    plot_all_condition_iv_curve(
                        curve_condition_values=self.vds,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        log_y=True,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
        ### New
        for param_name in key_params_names:
            param_values = episode.custom_data[param_name]

            avg_param_value = np.mean(param_values) if param_values else 0.0

            log_key = f"avg_{param_name}"

            metrics_logger.log_value(
                log_key,
                avg_param_value,
                reduce="mean",
            )
        return
