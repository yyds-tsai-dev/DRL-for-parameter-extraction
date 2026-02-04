import os
from datetime import datetime

import matplotlib.pyplot as plt
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
    # initial_colors = plt.get_cmap("Greens")(np.linspace(0.5, 1, num_curves))
    current_colors = plt.get_cmap("rainbow")(np.linspace(0.5, 1, num_curves))

    # === Iterate through each (Ugw, NOF) pair and plot with gradient colors ===
    for i, ugw_n in enumerate(curve_condition_values):
        label_target = "Experiments" if i == len(curve_condition_values) - 1 else None
        # label_initial = "Initial" if i == len(curve_condition_values) - 1 else None
        label_current = "Modeling" if i == len(curve_condition_values) - 1 else None
        # 1. Plot the target data (Measured) using the 'Blues' colormap.
        ax.plot(
            vgs,
            i_meas_dict[ugw_n],
            marker="o",
            linestyle="None",
            color=target_colors[i],
            label=label_target,
            ms=3,
        )

        # 2. Plot the initial simulation using the 'Greens' colormap.
        # ax.plot(
        #     vgs,
        #     i_sim_init_matrix[i, :],
        #     linestyle="--",  # Set line style to dashed
        #     color=initial_colors[i],
        #     label=label_initial,
        # )

        # 3. Plot the current simulation using the 'Reds' colormap.
        ax.plot(
            vgs,
            i_sim_current_matrix[i, :],
            linestyle="-",  # Set line style to solid
            color=current_colors[i],
            label=label_current,
        )

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
    ax.legend(loc="best")

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
        self.min_nrmse = 100.0

    def on_environment_created(
        self, *, env_runner, metrics_logger=None, env, env_context, **kwargs
    ):
        actual_env = env.envs[  # type: ignore
            0
        ].unwrapped  # type(actual_env).__name__ = EEHEMTEnv_Norm_Lgs

        if self.plot_data is None:
            print("\nFetching static plot data from the environment...\n")
            if hasattr(actual_env, "_get_plot_data_matrix"):
                # Fetch static plot data only once
                self.plot_data = actual_env._get_plot_data_matrix()
                if hasattr(actual_env, "curve_condition_values"):
                    self.curve_condition_values = actual_env.curve_condition_values
                if hasattr(actual_env, "vds"):
                    self.vds = actual_env.vds
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
            nrmse = last_info["nrmse"]
            if nrmse < self.min_nrmse:
                self.min_nrmse = nrmse
                
            metrics_logger.log_value(
                "min_nrmse",
                self.min_nrmse,
                reduce="mean",
            )
            print(f"\nFinal NRMSE: {nrmse:.4f}%\nMin NRMSE: {self.min_nrmse:.4f}%")

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
