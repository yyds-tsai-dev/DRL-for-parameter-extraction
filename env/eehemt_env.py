import json
import os

import gymnasium as gym
import numpy as np
import pandas as pd

# Only available on Linux with python 3.11
import verilogae  # type: ignore[import-untyped]
from dotenv import load_dotenv
from gymnasium.spaces import Box

from utils.dim_reduce import get_err_features
from utils.metrics import calculate_nrmse

load_dotenv()

current_dir = os.getcwd()
all_possible_key_params_path = os.path.join(
    current_dir, os.getenv("ALL_POSSIBLE_KEY_PARAMS_PATH", "")
)
# Dictionary of all possible key parameters
with open(all_possible_key_params_path, "r", encoding="utf-8") as f:
    ALL_POSSIBLE_KEY_PARAMS = json.load(f)

# Get key params name from environment variable
key_params_names = [
    name.strip() for name in os.getenv("KEY_PARAMS", "").split(",") if name.strip()
]
# Set key params config
key_params_config = {}
for name in key_params_names:
    if name in ALL_POSSIBLE_KEY_PARAMS:
        key_params_config[name] = ALL_POSSIBLE_KEY_PARAMS[name]
    else:
        print(
            f"Warning: Parameter '{name}' from environment variable not found in master config. Skipping."
        )
n_key_params = len(key_params_config)

CURVE_CONDITION_NAMES = os.getenv("CURVE_CONDITION_NAMES", "UGW,NOF").split(",")
TEMPERATURE = int(os.getenv("TEMPERATURE", 300))
EPSILON = float(os.getenv("EPSILON", 1e-15))


class EEHEMTEnv_Measure_VDS(gym.Env):
    """
    A custom Gymnasium environment for optimizing EE-HEMT model parameters.

    Attributes:
        action_space (gym.spaces.Box): The space of possible actions.
        observation_space (gym.spaces.Box): The space of possible observations.
        ...
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict) -> None:
        """
        Initializes the environment.

        Args:
            config (dict): A dictionary containing configuration parameters for the environment,
                           such as file paths and parameter tuning settings.
        """
        super(EEHEMTEnv_Measure_VDS, self).__init__()

        self.eehemt_model = verilogae.load(config.get("va_file_path", ""))  # type: ignore

        # === Vds & Vgs & Changeable Params ===
        self.csv_file_path = config.get("csv_file_path", "")
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(
                f"Measured data file not found:: {self.csv_file_path}"
            )
        measured_df = pd.read_csv(self.csv_file_path)
        self.vgs = measured_df["vg"].values
        self.n_vgs = len(self.vgs)
        print(f"==== Using Vgs values: {self.vgs} ====")
        self.vds = [float(col) for col in measured_df.columns if col != "vg"][1:11] + [
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
        ]
        # self.vds =  [
        #     1.0,
        #     1.5,
        #     2.0,
        #     2.5,
        #     3.0,
        #     3.5,
        #     4.0,
        #     4.5,
        # ]
        self.n_vds = len(self.vds)
        # print(
        #     f"==== Using Vds values: {', '.join(map(str, self.vds))} ===="
        # )
        print(
            f"==== Using {', '.join(CURVE_CONDITION_NAMES)} different values: {', '.join(map(str, self.vds))} ===="
        )

        # === Init & Target Params (Including key) Initialization ===
        self.random_init = config.get("random_init", False)
        print(
            f"\n==== Random initialization of parameters is {'enabled' if self.random_init else 'disabled'} ====\n"
        )
        self.init_params = {
            name: float(param.default)
            for name, param in self.eehemt_model.modelcard.items()
        }
        if key_params_config.get("DVcoVgo"):
            self.init_params["DVcoVgo"] = key_params_config["DVcoVgo"][
                "min"
            ]  # Initial DVcoVgo = 0.001V
        if key_params_config.get("DVgoVto"):
            self.init_params["DVgoVto"] = key_params_config["DVgoVto"][
                "min"
            ]  # Initial DVgoVto = 0.01V
        if key_params_config.get("DVtsoVto"):
            self.init_params["DVtsoVto"] = key_params_config["DVtsoVto"][
                "min"
            ]  # Initial DVtsoVto = 0.01V

        self.current_params = self.init_params.copy()
        self.KEY_PARAMS_MIN = np.array(
            [config["min"] for config in key_params_config.values()],
            dtype=np.float32,
        )
        self.KEY_PARAMS_MAX = np.array(
            [config["max"] for config in key_params_config.values()],
            dtype=np.float32,
        )

        # === Load I_meas (y_true) and sweep bias ===
        self.i_meas_dict = {}
        for vd in self.vds:
            filtered_data = measured_df[str(vd)].values
            if len(filtered_data) == self.n_vgs:
                self.i_meas_dict[vd] = filtered_data
        self.all_i_meas_matrix = np.array([self.i_meas_dict[vd] for vd in self.vds])
        self.all_i_meas_asinh_matrix = np.arcsinh(self.all_i_meas_matrix)
        self.add_log_err = config.get("add_log_err", False)
        print(
            f"\n==== Log error in observation is {'enabled' if self.add_log_err else 'disabled'} ====\n"
        )
        if self.add_log_err:
            self.all_i_meas_log_matrix = np.log10(self.all_i_meas_matrix + EPSILON)

        # === Action Space Definition ===
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(n_key_params,), dtype=np.float32
        )
        # self.action_space = Box(
        #     low=0.0, high=1.0, shape=(n_key_params,), dtype=np.float32
        # )
        # self.ACTION_FACTORS = np.array((config["max"] - config["min"]) * scaling_ratio
        #     [config["factor"] for config in key_params_config.values()],
        #     dtype=np.float32,
        # )  # Linear transform better than independent function transform
        self.ACTION_FACTORS = np.array(
            [
                (config["max"] - config["min"]) * 0.01
                for config in key_params_config.values()
            ],
            dtype=np.float32,
        )
        self.prev_params_delta = {name: EPSILON for name in key_params_names}

        # === Observation Space Definition ===
        # Observation space contains: [P_t, ΔP_{t-1}, E_t (error vector feature)]
        self.reduce_obs_err_dim = config.get("reduce_obs_err_dim", False)
        print(
            f"\n==== Reduce observation error dimension is {'enabled' if self.reduce_obs_err_dim else 'disabled'} ====\n"
        )
        param_low = [config["min"] for config in key_params_config.values()]
        param_high = [config["max"] for config in key_params_config.values()]

        prev_params_delta_low = -self.ACTION_FACTORS
        prev_params_delta_high = self.ACTION_FACTORS

        if self.reduce_obs_err_dim:
            total_err_len = int(os.getenv("N_FEATURES_PER_CURVE", 6)) * self.n_vds
        elif self.add_log_err:
            total_err_len = self.n_vgs * self.n_vds * 2  # 2 for linear & log error
        else:
            total_err_len = self.n_vgs * self.n_vds
        err_vector_low = np.full(total_err_len, -np.inf)
        err_vector_high = np.full(total_err_len, np.inf)

        low_bounds = np.concatenate(
            [param_low, prev_params_delta_low, err_vector_low]
        ).astype(np.float32)
        high_bounds = np.concatenate(
            [param_high, prev_params_delta_high, err_vector_high]
        ).astype(np.float32)
        # low_bounds = np.concatenate(
        #     [param_low, err_vector_low]
        # ).astype(np.float32)
        # high_bounds = np.concatenate(
        #     [param_high, err_vector_high]
        # ).astype(np.float32)
        self.observation_space = Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # === Episode Control ===
        self.MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS", 1000))
        self.REWARD_NORM_THRESHOLD = float(os.getenv("REWARD_NORM_THRESHOLD", 100.0))
        self.NRMSE_THRESHOLD = float(os.getenv("NRMSE_THRESHOLD", 80.0))
        self.current_step = 0

        # === Reward & Error Initialization ===
        # self.prev_nrmse = -1.0  # For reward calculation
        self.reward_norm = config.get("reward_norm", False)
        print(
            f"\n==== Reward normalization is {'enabled' if self.reward_norm else 'disabled'} ====\n"
        )

        if self.reward_norm:
            self.reward_running_mean = 0.0
            self.reward_running_var = 1.0
            self.reward_count = 0
            self.REWARD_ALPHA = float(
                os.getenv("REWARD_ALPHA", 0.01)
            )  # Running average decay factor
        self.huber_delta = float(os.getenv("HUBER_DELTA", 1.0))  # Huber loss delta

        # === External Resistance for IR Drop Correction ===
        # Internal Rs/Rd are modelcard params handled by the EEHEMT model itself.
        # Rs_ext/Rd_ext are purely external circuit resistances the model doesn't know about.
        self.Rs_ext = float(os.getenv("RS_EXT", 0.0))
        self.Rd_ext = float(os.getenv("RD_EXT", 0.0))
        self.ir_drop_n_iter = int(os.getenv("IR_DROP_N_ITER", 2))
        print(
            f"==== IR Drop: Rs_ext={self.Rs_ext} Ω, Rd_ext={self.Rd_ext} Ω, "
            f"n_iter={self.ir_drop_n_iter} ===="
        )

        # === Stagnation (停滯) detection settings ===
        # self.use_stagnation = config.get("use_stagnation", True)
        # if self.use_stagnation:
        #     print("\n==== Stagnation detection is enabled ====\n")
        #     self.STAGNATION_PATIENCE_STEPS = int(
        #         os.getenv("STAGNATION_PATIENCE_STEPS", 50)
        #     )  # step 耐心值
        #     self.STAGNATION_THRESHOLD = float(
        #         os.getenv("STAGNATION_THRESHOLD", 1e-3)
        #     )  # 進展的門檻
        #     self.stagnation_cnt = 0

    def _get_obs(self, concat_err_vector: np.ndarray) -> np.ndarray:
        """
        Constructs the observation vector for the agent.

        Observation = [P_t (current params), ΔP_{t-1} (previous param change), E_t (normalized error vector)]

        Returns:
            np.ndarray: The observation vector.
        """
        # 1. P_t: Current key params vector
        current_key_values = np.array(
            [self.current_params[name] for name in key_params_names],
            dtype=np.float32,
        )

        # 2. \Delta_P_{t-1}: Diff vector between current and previous params
        prev_params_delta = np.array(
            [self.prev_params_delta[name] for name in key_params_names],
            dtype=np.float32,
        )

        # 3. E_t: Error vector between I_meas and I_sim
        if self.reduce_obs_err_dim:
            err_features = get_err_features(
                self.vgs,  # type: ignore
                concat_err_vector,
                self.current_params["Vto"],
                self.current_params["Vgo"],
                self.n_vds,
            )
        else:
            err_features = concat_err_vector

        # 4. Combine observation vector
        obs = np.concatenate(
            [current_key_values, prev_params_delta, err_features]
        ).astype(np.float32)
        # obs = np.concatenate(
        #     [current_key_values, err_features]
        # ).astype(np.float32)
        # if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        #     print("Warning: NaN or Inf detected in obs, cleaning it.")
        #     obs = np.nan_to_num(obs, nan=0.0, posinf=1e5, neginf=-1e5)

        return obs

    def _get_info(self, nrmse: float) -> dict:
        """
        Generates the info dictionary returned at each step.

        Args:
            nrmse (float): The current NRSME value.
        """
        current_key_params = {
            name: self.current_params[name] for name in key_params_names
        }
        return {
            "nrmse": nrmse,
            "current_key_params": current_key_params,
        }

    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse transform function: converts normalized action [-1, 1] to actual parameter changes."""
        return action * self.ACTION_FACTORS

    def _run_all_curve_condition_sim(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper function to run simulations for all finger and width conditions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - A flattened numpy array containing all concatenated error vectors.
                - A numpy array of NRMSE values for each (Ugw, NOF) condition.
        """
        i_sim_results = []
        sim_params = {k: float(v) for k, v in self.current_params.items()}
        if sim_params.get("DVcoVgo"):
            sim_params.pop("DVcoVgo", None)
        if sim_params.get("DVgoVto"):
            sim_params.pop("DVgoVto", None)
        if sim_params.get("DVtsoVto"):
            sim_params.pop("DVtsoVto", None)
        # print(
        #     f"Step: {self.current_step}, Simulating with params: { {k: v for k, v in sim_params.items() if k in key_params_names} }"
        # )
        # Rs/Rd in va file only contribute I(d,di) and I(s,si) branch currents.
        # When calling eval() with intrinsic voltages (br_gisi/br_disi), the model
        # does NOT automatically apply Rs/Rd drops, so we must include them here.
        Rs_total = float(self.current_params["Rs"]) + self.Rs_ext
        Rd_total = float(self.current_params["Rd"]) + self.Rd_ext
        for vd_app in self.vds:
            # CURVE_CONDITION_NAMES[0] = "Vds" is a local analog variable in the VA file,
            # not a modelcard parameter, so it is not in sim_params and this line has no effect.
            # sim_params[CURVE_CONDITION_NAMES[0]] = vd_app

            # Vectorized fixed-point iteration for IR Drop correction.
            # R_tot = modelcard Rs/Rd (external-to-intrinsic) + circuit Rs_ext/Rd_ext
            I_est = np.zeros(self.n_vgs, dtype=np.float64)
            for _ in range(self.ir_drop_n_iter):
                vs_node = I_est * Rs_total  # shape: (n_vgs,)
                vd_node = vd_app - I_est * Rd_total  # shape: (n_vgs,)
                vgs_int = self.vgs - vs_node  # shape: (n_vgs,)
                vds_int = vd_node - vs_node  # shape: (n_vgs,)
                vgd_int = vgs_int - vds_int  # shape: (n_vgs,)
                I_est = np.asarray(
                    self.eehemt_model.functions["I_total"].eval(
                        temperature=TEMPERATURE,
                        voltages={
                            "br_gisi": vgs_int,
                            "br_disi": vds_int,
                            "br_gidi": vgd_int,
                        },
                        **sim_params,
                    ),
                ).ravel()

            i_sim_single_curve = I_est

            if np.any(np.isnan(i_sim_single_curve)) or np.any(
                np.isinf(i_sim_single_curve)
            ):
                i_sim_single_curve = np.nan_to_num(
                    i_sim_single_curve, nan=0.0, posinf=0.1, neginf=-0.1
                )

            i_sim_results.append(i_sim_single_curve)

        all_i_sim_matrix = np.array(i_sim_results)
        linear_err = self.all_i_meas_matrix - all_i_sim_matrix
        if self.add_log_err:
            log_err = self.all_i_meas_log_matrix - np.log10(all_i_sim_matrix + EPSILON)
            all_err_matrix = np.stack((linear_err, log_err), axis=-1)
        else:
            all_err_matrix = linear_err
        concat_err_vector = all_err_matrix.flatten().astype(np.float32)

        # Calculate NRMSE for each I-V curve (each row).
        nrmse_vals = np.array(
            [
                calculate_nrmse(i_meas_row, i_sim_row)
                for i_meas_row, i_sim_row in zip(
                    self.all_i_meas_matrix, all_i_sim_matrix
                )
            ],
            dtype=np.float32,
        )

        return all_i_sim_matrix, concat_err_vector, nrmse_vals

    def _update_reward_running_stats(self, reward: float):
        """
        Updates running mean and variance for reward normalization using exponential moving average.

        Args:
            reward (float): The raw reward value to incorporate into the stats.
        """
        if self.reward_count == 0:
            self.reward_running_mean = reward
            self.reward_running_var = 1.0
        else:
            delta = reward - self.reward_running_mean
            self.reward_running_mean += self.REWARD_ALPHA * delta
            self.reward_running_var = (
                1 - self.REWARD_ALPHA
            ) * self.reward_running_var + self.REWARD_ALPHA * delta * delta

        self.reward_count += 1

    def _normalize_reward(self, raw_reward: float) -> float:
        """
        Normalizes reward using running statistics.

        Args:
            raw_reward (float): The raw reward value.

        Returns:
            float: Normalized reward.
        """

        if not self.reward_norm:
            return raw_reward
        elif abs(raw_reward) > self.REWARD_NORM_THRESHOLD:
            # Update running statistics
            self._update_reward_running_stats(raw_reward)
            return raw_reward
        else:
            # Update running statistics
            self._update_reward_running_stats(raw_reward)

            # Normalize reward
            running_std = np.sqrt(self.reward_running_var) + EPSILON
            normalized_reward = (raw_reward - self.reward_running_mean) / running_std

            return normalized_reward

        # Optional: clip normalized reward to reasonable range
        # return np.clip(normalized_reward, -5.0, 5.0)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        """
        Resets the environment to its initial state for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        if options and "random_init" in options:
            self.random_init = options["random_init"]

        if self.random_init:
            for i, name in enumerate(key_params_names):
                self.current_params[name] = float(
                    np.random.uniform(self.KEY_PARAMS_MIN[i], self.KEY_PARAMS_MAX[i])
                )
                # print(f"DEBUG: {name} initialized to {self.current_params[name]}")
        else:
            self.current_params = self.init_params.copy()
        # Vgo = Vco - DVcoVgo
        if "Vco" in self.current_params and "DVcoVgo" in self.current_params:
            self.current_params["Vgo"] = (
                self.current_params["Vco"] - self.current_params["DVcoVgo"]
            )
        if "Vco" in self.current_params and "DVgoVto" in self.current_params:
            self.current_params["Vto"] = (
                self.current_params["Vco"]
                - self.current_params["DVcoVgo"]
                - self.current_params["DVgoVto"]
            )
        if "Vtso" in self.current_params and "DVtsoVto" in self.current_params:
            self.current_params["Vto"] = (
                self.current_params["Vtso"] - self.current_params["DVtsoVto"]
            )

        self.prev_params_delta = {name: EPSILON for name in key_params_names}

        self.current_step = 0
        # if self.use_stagnation:
        #     self.stagnation_cnt = 0

        # === Run initial simulation for all (Ugw, NOF) conditions & Calculate RMSPE ===
        _, init_err_vector, init_nrmse_vals = self._run_all_curve_condition_sim()
        # avg_init_rmspe = np.mean(init_rmspe_vals)
        ### New
        avg_init_nrmse = np.mean(init_nrmse_vals)
        # self.prev_nrmse = avg_init_nrmse

        observation = self._get_obs(init_err_vector)
        info = self._get_info(avg_init_nrmse)

        return observation, info

    def step(self, action: np.ndarray) -> tuple:
        """
        Executes one time step within the environment.

        This involves updating the model parameters based on the agent's action,
        simulating the I-V curve, calculating the new RMSPE, and determining the reward.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            tuple: A tuple containing the new observation, reward, terminated flag,
                   truncated flag, and info dictionary.
        """
        self.current_step += 1

        # === Update parameters and ensure they are within defined bounds ===
        key_params_delta = self._transform_action(action)
        for i, name in enumerate(key_params_names):
            self.current_params[name] += key_params_delta[i]
            # self.current_params[name] = self.KEY_PARAMS_MIN[i] + action[i] * (self.KEY_PARAMS_MAX[i] - self.KEY_PARAMS_MIN[i])

            self.current_params[name] = np.clip(
                self.current_params[name],
                self.KEY_PARAMS_MIN[i],
                self.KEY_PARAMS_MAX[i],
            )

        # Vgo = Vco - DVcoVgo
        if "Vco" in self.current_params and "DVcoVgo" in self.current_params:
            self.current_params["Vgo"] = (
                self.current_params["Vco"] - self.current_params["DVcoVgo"]
            )
        if "Vco" in self.current_params and "DVgoVto" in self.current_params:
            self.current_params["Vto"] = (
                self.current_params["Vco"]
                - self.current_params["DVcoVgo"]
                - self.current_params["DVgoVto"]
            )
        if "Vtso" in self.current_params and "DVtsoVto" in self.current_params:
            self.current_params["Vto"] = (
                self.current_params["Vtso"] - self.current_params["DVtsoVto"]
            )
        self.prev_params_delta = dict(zip(key_params_names, key_params_delta))

        # === Run simulations for all (Ugw, NOF) conditions ===
        all_i_sim_matrix, current_err_vector, nrmse_vals = (
            self._run_all_curve_condition_sim()
        )

        # === Calculate NRMSE for reward, termination conditions, and info ===
        current_nrmse = np.mean(nrmse_vals)
        # raw_reward = self.prev_nrmse - current_nrmse
        # raw_reward = np.clip(-np.log10((current_nrmse / 100.0) + EPSILON), -10.0, 10.0)

        diff = np.arcsinh(all_i_sim_matrix) - self.all_i_meas_asinh_matrix
        abs_diff = np.abs(diff)
        loss_linear = np.where(
            abs_diff <= self.huber_delta,
            0.5 * diff**2,
            self.huber_delta * (abs_diff - 0.5 * self.huber_delta),
        )
        # Log domain Huber loss (subthreshold)
        # log_sim = np.log10(np.abs(all_i_sim_matrix) + EPSILON)
        # log_meas = np.log10(np.abs(self.all_i_meas_matrix) + EPSILON)
        # diff_log = log_sim - log_meas
        # abs_diff_log = np.abs(diff_log)
        # loss_log = np.where(
        #     abs_diff_log <= self.huber_delta,
        #     0.5 * diff_log**2,
        #     self.huber_delta * (abs_diff_log - 0.5 * self.huber_delta),
        # )

        # log_loss_weight = float(os.getenv("LOG_LOSS_WEIGHT", 2.0))
        # combined_loss = np.mean(loss_linear) + log_loss_weight * np.mean(loss_log)

        raw_reward = np.clip(
            -np.log10(np.mean(loss_linear) + EPSILON) / 20.0,
            float(os.getenv("REWARD_MIN", -2.0)),
            float(os.getenv("REWARD_MAX", 2.0)),
        )  # Scale down for stability

        reward = self._normalize_reward(float(raw_reward))

        # self.prev_nrmse = current_nrmse

        # === Get the next observation and info ===
        observation = self._get_obs(current_err_vector)
        info = self._get_info(current_nrmse)

        # === Check Termination Conditions ===
        terminated_success = current_nrmse < self.NRMSE_THRESHOLD
        # if self.use_stagnation:
        #     if abs(reward) < self.STAGNATION_THRESHOLD:
        #         self.stagnation_cnt += 1
        #     else:
        #         self.stagnation_cnt = 0
        #     terminated_stagnation = (
        #         self.stagnation_cnt >= self.STAGNATION_PATIENCE_STEPS
        #     )
        #     terminated = terminated_success or terminated_stagnation
        # else:
        #     terminated = terminated_success
        terminated = terminated_success
        truncated = self.current_step >= self.MAX_EPISODE_STEPS

        if terminated_success:
            print(
                f"Success! NRMSE ({current_nrmse:.4f}) has reached the threshold ({self.NRMSE_THRESHOLD})."
            )
        # if self.use_stagnation and terminated_stagnation:
        #     print(
        #         f"Terminated due to stagnation ({self.STAGNATION_PATIENCE_STEPS} steps with little improvement)."
        #     )
        if truncated and not terminated:
            print("Reached maximum steps.")

        if terminated or truncated:
            info["i_sim_current_matrix"] = all_i_sim_matrix

        return observation, reward, terminated, truncated, info

    def _get_plot_data_matrix(self):
        return {
            "vgs": self.vgs,
            "i_meas_dict": self.i_meas_dict,
        }
