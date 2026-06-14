import os

import numpy as np

from env.eehemt_env import EEHEMTEnv_Measure_VDS
from evaluation.metrics import calculate_nrmse


def _env_config() -> dict[str, object]:
    cwd = os.getcwd()
    return {
        "va_file_path": os.path.join(cwd, os.getenv("VA_FILE_PATH", "")),
        "csv_file_path": os.path.join(cwd, os.getenv("CSV_FILE_PATH", "")),
        "random_init": False,
        "reduce_obs_err_dim": False,
        "reward_norm": False,
    }


def test_reset_observation_is_inside_declared_space():
    env = EEHEMTEnv_Measure_VDS(_env_config())

    observation, info = env.reset(seed=123)

    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert "arcsinh_huber_loss" in info
    assert "nrmse" in info
    assert info["nrmse"] == calculate_nrmse(
        env.all_i_meas_matrix,
        env.last_i_sim_current_matrix,
    )


def test_random_reduced_reset_observation_is_inside_declared_space():
    config = _env_config()
    config["random_init"] = True
    config["reduce_obs_err_dim"] = True
    env = EEHEMTEnv_Measure_VDS(config)

    observation, info = env.reset(seed=123)

    assert observation.dtype == np.float32
    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    assert "arcsinh_huber_loss" in info


def test_reward_uses_transformed_nrmse_objective(monkeypatch):
    monkeypatch.setenv("REWARD_MIN", "-5.0")
    monkeypatch.setenv("REWARD_MAX", "5.0")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert reward == np.clip(
        -np.log10((info["nrmse"] / 100.0) + 1e-15),
        -5.0,
        5.0,
    )
    assert "nrmse" in info
    assert info["nrmse"] == calculate_nrmse(
        env.all_i_meas_matrix,
        env.last_i_sim_current_matrix,
    )


def test_termination_uses_nrmse_threshold_not_arcsinh_huber_threshold(monkeypatch):
    monkeypatch.setenv("ARCSINH_HUBER_THRESHOLD", "0.0")
    monkeypatch.setenv("NRMSE_THRESHOLD", "999.0")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    env.reset(seed=123)

    _, _, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert info["nrmse"] < 999.0
    assert info["arcsinh_huber_loss"] > 0.0
    assert terminated is True
    assert truncated is False


def test_ir_drop_config_comes_from_env_config_not_only_environment(monkeypatch):
    monkeypatch.setenv("RS_EXT", "99.0")
    monkeypatch.setenv("RD_EXT", "99.0")
    monkeypatch.setenv("IR_DROP_N_ITER", "99")
    monkeypatch.setenv("IR_DROP_MAXFEV", "99")
    config = _env_config()
    config.update(
        {
            "rs_ext": 1.25,
            "rd_ext": 0.4,
            "ir_drop_n_iter": 3,
            "ir_drop_maxfev": 123,
        }
    )

    env = EEHEMTEnv_Measure_VDS(config)

    assert env.Rs_ext == 1.25
    assert env.Rd_ext == 0.4
    assert env.ir_drop_n_iter == 3
    assert env.ir_drop_maxfev == 123


def test_ir_drop_non_convergence_penalizes_and_truncates(monkeypatch):
    import env.parameter_flow as parameter_flow

    def fake_fsolve(*, func, x0, maxfev, full_output):
        return np.asarray(x0, dtype=float), {}, 2, "iteration is not making progress"

    monkeypatch.setattr(parameter_flow, "fsolve", fake_fsolve)
    monkeypatch.setenv("REWARD_MIN", "-5.0")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert reward == -5.0
    assert terminated is False
    assert truncated is True
    assert info["ir_drop_solver_converged"] is False
    assert len(info["ir_drop_solver_failures"]) == env.n_vds


def test_reset_initializes_episode_best_snapshot():
    env = EEHEMTEnv_Measure_VDS(_env_config())

    _, info = env.reset(seed=123)

    assert info["episode_best_nrmse"] == info["nrmse"]
    assert info["episode_best_arcsinh_huber_loss"] == info["arcsinh_huber_loss"]
    assert np.array_equal(
        info["episode_best_i_sim_current_matrix"],
        env.last_i_sim_current_matrix,
    )
    assert info["episode_best_key_params"] == info["current_key_params"]


def test_non_converged_reset_does_not_initialize_episode_best(monkeypatch):
    import env.parameter_flow as parameter_flow

    def fake_fsolve(*, func, x0, maxfev, full_output):
        return np.asarray(x0, dtype=float), {}, 2, "iteration is not making progress"

    env = EEHEMTEnv_Measure_VDS(_env_config())
    monkeypatch.setattr(parameter_flow, "fsolve", fake_fsolve)

    _, info = env.reset(seed=123)

    assert info["ir_drop_solver_converged"] is False
    assert "episode_best_nrmse" not in info
    assert "episode_best_i_sim_current_matrix" not in info


def test_non_terminal_step_info_omits_episode_best_matrix(monkeypatch):
    monkeypatch.setenv("MAX_EPISODE_STEPS", "2")
    monkeypatch.setenv("NRMSE_THRESHOLD", "0.0")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    env.reset(seed=123)

    _, _, terminated, truncated, step_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert terminated is False
    assert truncated is False
    assert "episode_best_i_sim_current_matrix" not in step_info


def test_episode_end_info_includes_best_snapshot(monkeypatch):
    monkeypatch.setenv("MAX_EPISODE_STEPS", "1")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    _, reset_info = env.reset(seed=123)

    _, _, terminated, truncated, step_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert terminated or truncated
    assert step_info["episode_best_nrmse"] <= max(
        reset_info["nrmse"],
        step_info["nrmse"],
    )
    assert "episode_best_i_sim_current_matrix" in step_info
    assert "episode_best_key_params" in step_info


def test_non_converged_candidate_does_not_update_episode_best():
    env = EEHEMTEnv_Measure_VDS(_env_config())
    env.reset(seed=123)
    previous_best_nrmse = env.episode_best_nrmse
    previous_best_loss = env.episode_best_arcsinh_huber_loss
    previous_best_matrix = np.array(env.episode_best_i_sim_current_matrix, copy=True)
    previous_best_params = dict(env.episode_best_key_params)
    synthetic_better_matrix = np.full_like(previous_best_matrix, 42.0)

    env._maybe_record_episode_best(
        arcsinh_huber_loss=previous_best_loss / 2.0,
        nrmse=previous_best_nrmse / 2.0,
        i_sim_current_matrix=synthetic_better_matrix,
        solver_converged=False,
    )

    assert env.episode_best_nrmse == previous_best_nrmse
    assert env.episode_best_arcsinh_huber_loss == previous_best_loss
    assert np.array_equal(env.episode_best_i_sim_current_matrix, previous_best_matrix)
    assert env.episode_best_key_params == previous_best_params
