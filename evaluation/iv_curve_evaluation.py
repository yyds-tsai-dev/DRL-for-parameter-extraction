from __future__ import annotations

import os
from datetime import date
from typing import Any

from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS

from utils.logging_config import get_logger
from utils.plot import save_evaluation_iv_curves

logger = get_logger(__name__)


def _episode_final_info(episode) -> dict:
    get_infos = getattr(episode, "get_infos", None)
    if callable(get_infos):
        try:
            info = get_infos(-1)
            return info or {}
        except (IndexError, TypeError, KeyError):
            return {}

    infos = getattr(episode, "infos", None)
    if infos is None:
        return {}

    try:
        return infos[-1] or {}
    except (IndexError, TypeError, KeyError):
        return {}


def _episode_env_steps(episode) -> int:
    value = getattr(episode, "env_steps", None)
    return int(value() if callable(value) else value or 0)


def _episode_agent_steps(episode) -> int:
    value = getattr(episode, "agent_steps", None)
    return int(value() if callable(value) else value or 0)


def _as_episode_list(sample_result: Any) -> list:
    if sample_result is None:
        return []
    if isinstance(sample_result, list):
        return sample_result
    if isinstance(sample_result, tuple):
        return list(sample_result)
    return [sample_result]


def _unwrap_env(candidate: Any) -> Any:
    env = getattr(candidate, "env", candidate)
    env = getattr(env, "unwrapped", env)
    envs = getattr(env, "envs", None)
    if envs:
        env = envs[0]
        env = getattr(env, "unwrapped", env)
    return env


def _static_plot_data_from_env_runner(env_runner: Any) -> tuple[list, dict] | None:
    env = _unwrap_env(env_runner)
    get_plot_data = getattr(env, "_get_plot_data_matrix", None)
    if not callable(get_plot_data):
        return None

    plot_data = get_plot_data()
    if not plot_data:
        return None

    curve_condition_values = getattr(env, "curve_condition_values", None)
    if curve_condition_values is None:
        curve_condition_values = getattr(env, "vds", None)
    if curve_condition_values is None:
        return None

    return list(curve_condition_values), dict(plot_data)


def _remote_eval_worker_ids(eval_workers: Any) -> list:
    get_healthy_worker_ids = getattr(eval_workers, "healthy_worker_ids", None)
    if callable(get_healthy_worker_ids):
        return list(get_healthy_worker_ids())

    worker_manager = getattr(eval_workers, "_worker_manager", None)
    get_actor_ids = getattr(worker_manager, "actor_ids", None)
    if callable(get_actor_ids):
        return list(get_actor_ids())

    return []


def _foreach_one_eval_runner(eval_workers: Any, func) -> list:
    actor_ids = _remote_eval_worker_ids(eval_workers)
    if actor_ids:
        return eval_workers.foreach_env_runner(
            func=func,
            local_env_runner=False,
            remote_worker_ids=[actor_ids[0]],
        )

    return eval_workers.foreach_env_runner(
        func=func,
        local_env_runner=True,
    )


def _collect_static_plot_data(eval_workers: Any) -> tuple[list, dict] | None:
    foreach_env_runner = getattr(eval_workers, "foreach_env_runner", None)
    if callable(foreach_env_runner):
        results = _foreach_one_eval_runner(eval_workers, _static_plot_data_from_env_runner)
        for result in results:
            if result is not None:
                return result

    return _static_plot_data_from_env_runner(eval_workers)


def _sample_one_eval_runner(eval_workers: Any) -> list:
    def sample_and_get_metrics(worker):
        return (
            worker.sample(num_episodes=1),
            worker.get_metrics(),
        )

    return _foreach_one_eval_runner(eval_workers, sample_and_get_metrics)


def _plot_dir() -> str:
    base_dir = os.getenv("PLOT_DIR", os.path.join("result", "iv-curve"))
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.getenv("PROJECT_ROOT", os.getcwd()), base_dir)
    algo_name = os.getenv("ALGO_NAME", "ppo")
    return os.path.join(base_dir, algo_name, date.today().isoformat())


def _next_evaluation_index(algorithm: Any) -> int:
    evaluation_index = int(getattr(algorithm, "_iv_curve_evaluation_index", 0)) + 1
    setattr(algorithm, "_iv_curve_evaluation_index", evaluation_index)
    return evaluation_index


def evaluate_and_plot_iv_curve(algorithm, eval_workers):
    """RLlib custom evaluation hook that plots the sampled I-V curve on the driver."""
    sample_and_metrics = _sample_one_eval_runner(eval_workers)

    episodes = []
    env_runner_metrics = []
    for sample_result, metrics in sample_and_metrics:
        episodes.extend(_as_episode_list(sample_result))
        env_runner_metrics.append(metrics)

    metric_key = (EVALUATION_RESULTS, ENV_RUNNER_RESULTS)
    algorithm.metrics.aggregate(env_runner_metrics, key=metric_key)
    eval_results = algorithm.metrics.peek(metric_key)

    env_steps = sum(_episode_env_steps(episode) for episode in episodes)
    agent_steps = sum(_episode_agent_steps(episode) for episode in episodes)
    final_info = _episode_final_info(episodes[0]) if episodes else {}
    static_plot_data = _collect_static_plot_data(eval_workers)
    if static_plot_data is None:
        logger.warning(
            "Skipping evaluation I-V curve plot because static plot data could not "
            "be collected from evaluation workers."
        )
        return eval_results, env_steps, agent_steps

    curve_condition_values, plot_data = static_plot_data
    i_sim_current_matrix = final_info.get("i_sim_current_matrix")
    if i_sim_current_matrix is None:
        i_sim_current_matrix = plot_data.get("i_sim_current_matrix")
    if i_sim_current_matrix is None:
        logger.warning(
            "Skipping evaluation I-V curve plot because neither final episode info "
            "nor the evaluation environment has i_sim_current_matrix."
        )
        return eval_results, env_steps, agent_steps

    plot_data["i_sim_current_matrix"] = i_sim_current_matrix
    evaluation_index = _next_evaluation_index(algorithm)
    training_iteration = int(getattr(algorithm, "iteration", 0) or 0)
    fit_loss = final_info.get("arcsinh_huber_loss")

    try:
        save_evaluation_iv_curves(
            curve_condition_values=curve_condition_values,
            plot_data=plot_data,
            plot_dir=_plot_dir(),
            evaluation_index=evaluation_index,
            training_iteration=training_iteration,
            fit_loss=fit_loss,
        )
    except Exception:
        logger.exception(
            "Failed to save evaluation I-V curve plot "
            "(evaluation_index=%s, training_iteration=%s, fit_loss=%s).",
            evaluation_index,
            training_iteration,
            fit_loss,
        )
        raise

    return eval_results, env_steps, agent_steps
