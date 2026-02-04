import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from env.eehemt_env import CHANGE_PARAM_NAMES, key_params_names

load_dotenv()
### New
PLOT_PERIOD = int(os.getenv("PLOT_PERIOD", 5))


def plot_iv_curve(
    plot_data: dict,
    plot_initial: bool = True,
    plot_modified: bool = True,
    plot_current: bool = True,
    save_path: str | None = None,
):
    """
    Use pre-calculated data to plot the I-V curve.
    """
    output_dir = os.path.dirname(save_path) if save_path else "results"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Plot measured data from the dictionary
    plt.plot(
        plot_data["vgs"], plot_data["i_meas"], "ko", label="Measured Data (Target)"
    )

    # Plot simulated curves based on options
    if plot_initial:
        plt.plot(
            plot_data["vgs"],
            plot_data["i_sim_initial"],
            "b--",
            label=f"Simulated (Initial Params, Vto={plot_data['vto']:.2f})",
        )

    if plot_modified:
        plt.plot(
            plot_data["vgs"],
            plot_data["i_sim_modified"],
            "g-.",
            label=f"Simulated (Modified Initial Params, Vto={plot_data['vto']:.2f})",
        )

    if plot_current:
        plt.plot(
            plot_data["vgs"],
            plot_data["i_sim_current"],
            "r-",
            label=f"Simulated (Current Params, Vto={plot_data['vto']:.2f})",
        )

    # Style the plot
    plt.title("I-V Curve Comparison")
    plt.xlabel("Gate Voltage (Vg) [V]")
    plt.ylabel("Log Drain Current (Id) [A]")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"I-V curve plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_all_lg_iv_curve(
    lg_values: list,
    plot_data: dict,
    plot_dir: str,
):
    """
    Plots and saves individual I-V curves for each lg condition.

    Args:
        lg_values (list): A list containing all lg float values.
        plot_data (dict): A dictionary containing static plotting data,
                          such as 'vgs', 'i_meas_dict', etc.
        plot_dir (str): The directory path to save the plots.
    """
    # Get static data from plot_data
    vgs = plot_data["vgs"]
    i_meas_dict = plot_data["i_meas_dict"]
    i_sim_init_matrix = plot_data["i_sim_init_matrix"]
    # i_sim_modified_matrix = plot_data["i_sim_modified_matrix"]
    i_sim_current_matrix = plot_data["i_sim_current_matrix"]

    plt.figure(figsize=(12, 8))

    # Iterate through each lg and its corresponding index
    for i, lg in enumerate(lg_values):
        # 1. Plot the target data (Measured)
        plt.plot(vgs, i_meas_dict[lg], "o", label=f"Target (lg={lg})")

        # 2. Plot the simulated curve with initial parameters
        plt.plot(
            vgs,
            i_sim_init_matrix[i, :],  # Get the i-th row
            "b--",
            label=f"Initial (lg={lg})",
        )

        # 3. Plot the simulated curve with the agent's final parameters
        plt.plot(
            vgs,
            i_sim_current_matrix[i, :],  # Get the i-th row
            "r-",
            label=f"Current (lg={lg})",
        )

    # Set the plot style
    # plt.title(f"I-V Curve Comparison (lg = {lg:.2f})")

    plt.title("I-V Curve Comparison for All lg Values")
    plt.xlabel("Gate Voltage (Vg) [V]")
    plt.ylabel("Log Drain Current (Id) [A]")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # type: ignore
    # Save the plot
    save_path = os.path.join(plot_dir, "final_iv_curve_all_lg.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"==== I-V curve plot saved in {save_path} ====")


def plot_all_condition_iv_curve(
    ugw_n_values: list,
    plot_data: dict,
    plot_dir: str,
    log_y: bool = True,
    plot_cnt: int = 0,
):
    """
    Plots and saves I-V curves for all (Ugw, NOF) conditions on a single graph.
    Each curve type (Target, Initial, Current) has its own color gradient.

    Args:
        ugw_n_values (list): A list containing all (Ugw, NOF) float values.
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
    num_curves = len(ugw_n_values)
    target_colors = plt.get_cmap("Blues")(np.linspace(0.5, 1, num_curves))
    # initial_colors = plt.get_cmap("Greens")(np.linspace(0.5, 1, num_curves))
    current_colors = plt.get_cmap("Reds")(np.linspace(0.5, 1, num_curves))

    # === Iterate through each (Ugw, NOF) pair and plot with gradient colors ===
    for i, ugw_n in enumerate(ugw_n_values):
        label_target = "Target" if i == len(ugw_n_values) - 1 else None
        # label_initial = "Initial" if i == len(ugw_n_values) - 1 else None
        label_current = "PPO" if i == len(ugw_n_values) - 1 else None
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
    ax.set_title(f"I-V Curve Comparison for All {', '.join(CHANGE_PARAM_NAMES)} Values")
    ax.set_xlabel("Gate Voltage (Vg) [V]")
    if log_y:
        ax.set_ylabel("Log Drain Current (Id) [mA]")
        ax.set_yscale("log")
        save_path = os.path.join(
            plot_dir, f"iv_curve_all_{'_'.join(CHANGE_PARAM_NAMES)}_log_{plot_cnt}.png"
        )
    else:
        ax.set_ylabel("Drain Current (Id) [mA]")
        save_path = os.path.join(
            plot_dir, f"iv_curve_all_{'_'.join(CHANGE_PARAM_NAMES)}_{plot_cnt}.png"
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
        self.ugw_n_values = None  # Store lg values for plotting
        self.vds_values = None
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
                if hasattr(actual_env, "ugw_n_values"):
                    self.ugw_n_values = actual_env.ugw_n_values
                if hasattr(actual_env, "vds_values"):
                    self.vds_values = actual_env.vds_values
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

            self.plot_data["i_sim_current_matrix"] = last_info["i_sim_current_matrix"]  # type: ignore

            print(f"\nFinal NRMSE: {nrmse:.5f}\nMin NRMSE: {self.min_nrmse:.5f}")

            ### New
            if self.plot_cnt % PLOT_PERIOD == 0:
                if self.ugw_n_values is not None:
                    plot_all_condition_iv_curve(
                        ugw_n_values=self.ugw_n_values,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        # log_y=os.getenv("LOG_Y", "True").lower() == "true",
                        log_y=False,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
                    plot_all_condition_iv_curve(
                        ugw_n_values=self.ugw_n_values,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        log_y=True,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
                elif self.vds_values is not None:
                    plot_all_condition_iv_curve(
                        ugw_n_values=self.vds_values,
                        plot_data=self.plot_data,  # type: ignore
                        plot_dir=self.plot_dir,
                        # log_y=os.getenv("LOG_Y", "True").lower() == "true",
                        log_y=False,
                        plot_cnt=self.plot_cnt // PLOT_PERIOD,
                    )
                    plot_all_condition_iv_curve(
                        ugw_n_values=self.vds_values,
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
