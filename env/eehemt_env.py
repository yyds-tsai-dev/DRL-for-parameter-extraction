import json
import os

import gymnasium as gym
import numpy as np

from dotenv import load_dotenv
from gymnasium.spaces import Box

from env.parameter_flow import (
    ArcsinhHuberMetric,
    EEHEMTSimulator,
    MeasuredCurveDataset,
    ParameterSpecCollection,
)
from utils.dim_reduce import get_err_features

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
PARAMETER_SPECS = ParameterSpecCollection.from_config(
    ALL_POSSIBLE_KEY_PARAMS, key_params_names
)
key_params_names = PARAMETER_SPECS.names
n_key_params = len(PARAMETER_SPECS)

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

        # === Vds & Vgs & Changeable Params ===
        self.va_file_path = config.get("va_file_path", "")
        self.csv_file_path = config.get("csv_file_path", "")
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(
                f"Measured data file not found:: {self.csv_file_path}"
            )
        self.dataset = MeasuredCurveDataset.from_csv(self.csv_file_path)
        self.vgs = self.dataset.vgs
        self.n_vgs = len(self.vgs)
        print(f"==== Using Vgs values: {self.vgs} ====")
        self.vds = self.dataset.vds
        self.n_vds = len(self.vds)
        print(
            f"==== Using {', '.join(CURVE_CONDITION_NAMES)} different values: {', '.join(map(str, self.vds))} ===="
        )

        # === External Resistance for IR Drop Correction ===
        # Internal Rs/Rd are modelcard params handled by the EEHEMT model itself.
        # Rs_ext/Rd_ext are purely external circuit resistances the model doesn't know about.
        self.Rs_ext = float(os.getenv("RS_EXT", 0.0))
        self.Rd_ext = float(os.getenv("RD_EXT", 0.0))
        self.ir_drop_n_iter = int(os.getenv("IR_DROP_N_ITER", 2))
        self.ir_drop_maxfev = int(os.getenv("IR_DROP_MAXFEV", 40))
        print(
            f"==== IR Drop: Rs_ext={self.Rs_ext} Ω, Rd_ext={self.Rd_ext} Ω, "
            f"n_iter={self.ir_drop_n_iter}, maxfev={self.ir_drop_maxfev} ===="
        )
        self.simulator = EEHEMTSimulator.from_va_file(
            self.va_file_path,
            temperature=TEMPERATURE,
            rs_ext=self.Rs_ext,
            rd_ext=self.Rd_ext,
            ir_drop_n_iter=self.ir_drop_n_iter,
            ir_drop_maxfev=self.ir_drop_maxfev,
        )

        # === Init Params (Including derived control params) ===
        self.random_init = config.get("random_init", False)
        print(
            f"\n==== Random initialization of parameters is {'enabled' if self.random_init else 'disabled'} ====\n"
        )
        self.init_params = PARAMETER_SPECS.normalize_params(
            self.simulator.modelcard_defaults()
        )
        self.current_params = self.init_params.copy()
        self.KEY_PARAMS_MIN = PARAMETER_SPECS.min_values
        self.KEY_PARAMS_MAX = PARAMETER_SPECS.max_values

        # === Load I_meas (y_true) and sweep bias ===
        self.i_meas_dict = self.dataset.current_by_vds
        self.all_i_meas_matrix = self.dataset.current_matrix

        # === Action Space Definition ===
        self.action_space = Box(
            low=np.full(n_key_params, -1.0, dtype=np.float32),
            high=np.full(n_key_params, 1.0, dtype=np.float32),
            dtype=np.float32,
        )
        # self.action_space = Box(
        #     low=0.0, high=1.0, shape=(n_key_params,), dtype=np.float32
        # )
        # self.ACTION_FACTORS = np.array((config["max"] - config["min"]) * scaling_ratio
        #     [config["factor"] for config in key_params_config.values()],
        #     dtype=np.float32,
        # )  # Linear transform better than independent function transform
        self.ACTION_FACTORS = PARAMETER_SPECS.action_factors(range_fraction=0.01)
        self.prev_params_delta = {name: EPSILON for name in key_params_names}

        # === Observation Space Definition ===
        # Observation space contains: [P_t, ΔP_{t-1}, E_t (error vector feature)]
        self.reduce_obs_err_dim = config.get("reduce_obs_err_dim", False)
        print(
            f"\n==== Reduce observation error dimension is {'enabled' if self.reduce_obs_err_dim else 'disabled'} ====\n"
        )
        param_low = self.KEY_PARAMS_MIN
        param_high = self.KEY_PARAMS_MAX

        prev_params_delta_low = -self.ACTION_FACTORS
        prev_params_delta_high = self.ACTION_FACTORS
        self.OBS_ERR_BOUND = float(os.getenv("OBS_ERR_BOUND", 1e6))

        if self.reduce_obs_err_dim:
            total_err_len = int(os.getenv("N_FEATURES_PER_CURVE", 6)) * self.n_vds
        else:
            total_err_len = self.n_vgs * self.n_vds
        err_vector_low = np.full(total_err_len, -self.OBS_ERR_BOUND, dtype=np.float32)
        err_vector_high = np.full(total_err_len, self.OBS_ERR_BOUND, dtype=np.float32)

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
        self.ARCSINH_HUBER_THRESHOLD = float(
            os.getenv("ARCSINH_HUBER_THRESHOLD", 1e-5)
        )
        self.REWARD_MIN = float(os.getenv("REWARD_MIN", -5.0))
        self.REWARD_MAX = float(os.getenv("REWARD_MAX", 5.0))
        self.current_step = 0

        # === Reward & Error Initialization ===
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
        self.metric = ArcsinhHuberMetric(
            delta=self.huber_delta,
            epsilon=EPSILON,
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

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=self.OBS_ERR_BOUND,
            neginf=-self.OBS_ERR_BOUND,
        ).astype(np.float32)
        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

    def _get_info(self, arcsinh_huber_loss: float) -> dict:
        """
        Generates the info dictionary returned at each step.

        Args:
            arcsinh_huber_loss (float): The current fit loss.
        """
        current_key_params = {
            name: self.current_params[name] for name in key_params_names
        }
        return {
            "arcsinh_huber_loss": arcsinh_huber_loss,
            "current_key_params": current_key_params,
        }

    def _transform_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse transform function: converts normalized action [-1, 1] to actual parameter changes."""
        return action * self.ACTION_FACTORS

    def _run_all_curve_condition_sim(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper function to run simulations for all Vds conditions.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - A flattened numpy array containing all concatenated error vectors.
                - A numpy array of arcsinh Huber losses for each Vds condition.
        """
        self.current_params = PARAMETER_SPECS.normalize_params(self.current_params)
        all_i_sim_matrix = self.simulator.simulate_current_matrix(
            params=self.current_params,
            vgs=self.vgs,
            vds_values=self.vds,
            current_step=self.current_step,
        )
        linear_err = self.all_i_meas_matrix - all_i_sim_matrix
        concat_err_vector = linear_err.flatten().astype(np.float32)
        fit_loss_vals = self.metric.per_curve_loss(
            self.all_i_meas_matrix, all_i_sim_matrix
        )

        return all_i_sim_matrix, concat_err_vector, fit_loss_vals

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
                    self.np_random.uniform(self.KEY_PARAMS_MIN[i], self.KEY_PARAMS_MAX[i])
                )
                # print(f"DEBUG: {name} initialized to {self.current_params[name]}")
        else:
            self.current_params = self.init_params.copy()
        self.current_params = PARAMETER_SPECS.normalize_params(self.current_params)

        self.prev_params_delta = {name: EPSILON for name in key_params_names}

        self.current_step = 0
        # if self.use_stagnation:
        #     self.stagnation_cnt = 0

        # === Run initial simulation for all Vds conditions & calculate fit loss ===
        _, init_err_vector, init_loss_vals = self._run_all_curve_condition_sim()
        avg_init_loss = float(np.mean(init_loss_vals))

        observation = self._get_obs(init_err_vector)
        info = self._get_info(avg_init_loss)

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
        self.current_params, actual_delta = PARAMETER_SPECS.apply_delta(
            self.current_params, key_params_delta
        )
        self.prev_params_delta = actual_delta

        # === Run simulations for all (Ugw, NOF) conditions ===
        all_i_sim_matrix, current_err_vector, fit_loss_vals = (
            self._run_all_curve_condition_sim()
        )

        # === Calculate arcsinh Huber loss for reward, termination, and info ===
        current_loss = float(np.mean(fit_loss_vals))
        raw_reward = self.metric.scaled_reward_from_loss(
            current_loss,
            reward_min=self.REWARD_MIN,
            reward_max=self.REWARD_MAX,
        )
        reward = self._normalize_reward(float(raw_reward))

        # === Get the next observation and info ===
        observation = self._get_obs(current_err_vector)
        info = self._get_info(current_loss)

        # === Check Termination Conditions ===
        terminated_success = current_loss < self.ARCSINH_HUBER_THRESHOLD
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
                "Success! Arcsinh Huber loss "
                f"({current_loss:.6g}) has reached the threshold "
                f"({self.ARCSINH_HUBER_THRESHOLD:.6g})."
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
