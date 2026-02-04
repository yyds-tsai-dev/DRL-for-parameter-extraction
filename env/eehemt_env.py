import os
import json

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
all_possible_key_params_path = os.path.join(current_dir, os.getenv("ALL_POSSIBLE_KEY_PARAMS_PATH", ""))
# Dictionary of all possible key parameters
with open(all_possible_key_params_path, 'r', encoding='utf-8') as f:
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
EPSILON = float(os.getenv("EPSILON", 1e-9))


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
        self.vgs = measured_df['vg'].values
        self.n_vgs = len(self.vgs)
        print(f"==== Using Vgs values: {self.vgs} ====")
        self.vds = [float(col) for col in measured_df.columns if col != 'vg']
        self.n_vds = len(self.vds)
        # print(
        #     f"==== Using Vds values: {', '.join(map(str, self.vds))} ===="
        # )
        print(
            f"==== Using {', '.join(CURVE_CONDITION_NAMES)} different values: {', '.join(map(str,self.vds))} ===="
        )

        # === Init & Target Params (Including key) Initialization ===
        self.init_params = {
            name: float(param.default) for name, param in self.eehemt_model.modelcard.items()
        }
        for name in key_params_names:
            min_val = key_params_config[name]["min"]
            max_val = key_params_config[name]["max"]
            self.init_params[name] = float(np.random.uniform(min_val, max_val))
            print(f"==== {name} initialized to random value: {self.init_params[name]} ====")

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

        # === Action Space Definition ===
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(n_key_params,), dtype=np.float32
        )
        self.ACTION_FACTORS = np.array(
            [config["factor"] for config in key_params_config.values()],
            dtype=np.float32,
        )  # Linear transform better than independent function transform
        self.prev_params_delta = {name: EPSILON for name in key_params_names}

        # === Observation Space Definition ===
        # Observation space contains: [P_t, ΔP_{t-1}, E_t (error vector feature)]
        self.reduce_obs_err_dim = config.get("reduce_obs_err_dim", True)
        print(
            f"\n==== Reduce observation error dimension is {'enabled' if self.reduce_obs_err_dim else 'disabled'} ====\n"
        )
        param_low = [config["min"] for config in key_params_config.values()]
        param_high = [config["max"] for config in key_params_config.values()]

        prev_params_delta_low = -self.ACTION_FACTORS
        prev_params_delta_high = self.ACTION_FACTORS

        if self.reduce_obs_err_dim:
            total_err_len = int(os.getenv("N_FEATURES_PER_CURVE", 6)) * self.n_vds
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
        self.observation_space = Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        # === Episode Control ===
        self.MAX_EPISODE_STEPS = int(os.getenv("MAX_EPISODE_STEPS", 1000))
        self.REWARD_NORM_THRESHOLD = float(os.getenv("REWARD_NORM_THRESHOLD", 100.0))
        self.NRMSE_THRESHOLD = float(os.getenv("NRMSE_THRESHOLD", 80.0))
        self.current_step = 0

        # === Error Initialization ===
        self.prev_nrmse = -1.0  # For reward calculation
        self.reward_norm = config.get("reward_norm", True)
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
        all_i_meas_matrix = np.array(
            [self.i_meas_dict[vd] for vd in self.vds]
        )

        i_sim_results = []
        sim_params = self.current_params.copy()
        sim_params.pop("DVcoVgo", None)
        for vd in self.vds:
            current_vds_vector = np.full_like(self.vgs, vd)
            current_sweep_bias = {
                "br_gisi": self.vgs,
                "br_disi": current_vds_vector
            }
            
            sim_params[CURVE_CONDITION_NAMES[0]] = vd
            i_sim_single_curve = self.eehemt_model.functions["Ids"].eval(
                temperature=TEMPERATURE,
                voltages=current_sweep_bias ,
                **sim_params,
            )

            if np.any(np.isnan(i_sim_single_curve)) or np.any(
                np.isinf(i_sim_single_curve)
            ):
                i_sim_single_curve = np.nan_to_num(
                    i_sim_single_curve, nan=0.0, posinf=0.1, neginf=-0.1
                )

            i_sim_results.append(i_sim_single_curve)
        
        all_i_sim_matrix = np.array(i_sim_results)
        all_err_matrix = all_i_meas_matrix - all_i_sim_matrix
        concat_err_vector = all_err_matrix.flatten().astype(np.float32)

        # Calculate NRMSE for each I-V curve (each row).
        nrmse_vals = np.array(
            [
                calculate_nrmse(i_meas_row, i_sim_row)
                for i_meas_row, i_sim_row in zip(all_i_meas_matrix, all_i_sim_matrix)
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
        self.current_params = self.init_params.copy()
        # Vgo = Vco - DVcoVgo
        if "Vco" in self.current_params and "DVcoVgo" in self.current_params:
            self.current_params["Vgo"] = self.current_params["Vco"] - self.current_params["DVcoVgo"]

        self.prev_params_delta = {name: EPSILON for name in key_params_names}

        self.current_step = 0
        self.stagnation_cnt = 0

        # === Run initial simulation for all (Ugw, NOF) conditions & Calculate RMSPE ===
        _, init_err_vector, init_nrmse_vals = self._run_all_curve_condition_sim()
        # avg_init_rmspe = np.mean(init_rmspe_vals)
        ### New
        avg_init_nrmse = np.mean(init_nrmse_vals)
        self.prev_nrmse = avg_init_nrmse

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

            self.current_params[name] = np.clip(
                self.current_params[name],
                self.KEY_PARAMS_MIN[i],
                self.KEY_PARAMS_MAX[i],
            )

        # Vgo = Vco - DVcoVgo
        if "Vco" in self.current_params and "DVcoVgo" in self.current_params:
            self.current_params["Vgo"] = self.current_params["Vco"] - self.current_params["DVcoVgo"]
        self.prev_params_delta = dict(zip(key_params_names, key_params_delta))

        # === Run simulations for all (Ugw, NOF) conditions ===
        all_i_sim_matrix, current_err_vector, nrmse_vals = self._run_all_curve_condition_sim()

        # === Calculate NRMSE for reward, termination conditions, and info ===
        ### New
        current_nrmse = np.mean(nrmse_vals)
        ### New
        # raw_reward = self.prev_nrmse - current_nrmse
        raw_reward = np.clip(-np.log10(current_nrmse + EPSILON), -10.0, 10.0)
        reward = self._normalize_reward(float(raw_reward))

        self.prev_nrmse = current_nrmse

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

