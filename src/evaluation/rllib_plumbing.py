"""RLlib new-API-stack evaluation plumbing shared by all problems' eval hooks.

These helpers are the mechanics of sampling exactly one evaluation episode
from the evaluation EnvRunnerGroup and reading episode step counts. They are
problem-agnostic; artifact rendering stays in each problem's evaluation
module.
"""

from __future__ import annotations

from typing import Any


def as_episode_list(sample_result: Any) -> list[Any]:
    if sample_result is None:
        return []
    if isinstance(sample_result, list):
        return sample_result
    if isinstance(sample_result, tuple):
        return list(sample_result)
    return [sample_result]


def episode_env_steps(episode: Any) -> int:
    value = getattr(episode, "env_steps", None)
    return int(value() if callable(value) else value or 0)


def episode_agent_steps(episode: Any) -> int:
    value = getattr(episode, "agent_steps", None)
    return int(value() if callable(value) else value or 0)


def remote_eval_worker_ids(eval_workers: Any) -> list[Any]:
    get_healthy_worker_ids = getattr(eval_workers, "healthy_worker_ids", None)
    if callable(get_healthy_worker_ids):
        return list(get_healthy_worker_ids())

    worker_manager = getattr(eval_workers, "_worker_manager", None)
    get_actor_ids = getattr(worker_manager, "actor_ids", None)
    if callable(get_actor_ids):
        return list(get_actor_ids())

    return []


def foreach_one_eval_runner(eval_workers: Any, func) -> list[Any]:
    actor_ids = remote_eval_worker_ids(eval_workers)
    foreach_env_runner = getattr(eval_workers, "foreach_env_runner")
    if actor_ids:
        return foreach_env_runner(
            func=func,
            local_env_runner=False,
            remote_worker_ids=[actor_ids[0]],
        )

    return foreach_env_runner(
        func=func,
        local_env_runner=True,
    )


def sample_one_eval_runner(eval_workers: Any) -> list[Any]:
    def sample_and_get_metrics(worker):
        return worker.sample(num_episodes=1), worker.get_metrics()

    return foreach_one_eval_runner(eval_workers, sample_and_get_metrics)
