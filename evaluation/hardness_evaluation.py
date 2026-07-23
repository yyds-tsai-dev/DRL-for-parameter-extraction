from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any

import pandas as pd
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS


@dataclass(frozen=True)
class HardnessEvaluationPaths:
    csv_path: Path
    json_path: Path


def _sanitize_for_artifact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sanitize_for_artifact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_artifact(item) for item in value]
    if hasattr(value, "tolist"):
        return _sanitize_for_artifact(value.tolist())
    if hasattr(value, "item"):
        return _sanitize_for_artifact(value.item())
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            return None
    return value


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(record)
    best_composition = flattened.pop("best_composition", {}) or {}
    for key, value in best_composition.items():
        flattened[f"best_{key}"] = value
    return flattened


def _default_output_dir() -> Path:
    base_dir = Path(os.getenv("PROJECT_ROOT") or Path.cwd())
    return base_dir / "result" / "hardness-evaluation" / date.today().isoformat()


def save_hardness_evaluation(
    record: dict[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> HardnessEvaluationPaths:
    output_path = Path(output_dir) if output_dir is not None else _default_output_dir()
    output_path.mkdir(parents=True, exist_ok=True)
    sanitized_record = _sanitize_for_artifact(record)

    evaluation_index = int(sanitized_record["evaluation_index"])
    training_iteration = int(sanitized_record["training_iteration"])
    stem = f"eval_{evaluation_index:06d}_iter_{training_iteration:06d}"
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"

    pd.DataFrame([_flatten_record(sanitized_record)]).to_csv(csv_path, index=False)
    json_payload = json.dumps(
        sanitized_record,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    json_path.write_text(json_payload, encoding="utf-8")
    return HardnessEvaluationPaths(csv_path=csv_path, json_path=json_path)


def _as_episode_list(sample_result: Any) -> list[Any]:
    if sample_result is None:
        return []
    if isinstance(sample_result, list):
        return sample_result
    if isinstance(sample_result, tuple):
        return list(sample_result)
    return [sample_result]


def _episode_final_info(episode: Any) -> dict[str, Any]:
    get_infos = getattr(episode, "get_infos", None)
    if callable(get_infos):
        try:
            info = get_infos(-1)
            if isinstance(info, dict):
                return info
        except (IndexError, TypeError, KeyError):
            pass

    infos = getattr(episode, "infos", None)
    if infos is None:
        return {}

    try:
        info = infos[-1] or {}
    except (IndexError, TypeError, KeyError):
        return {}
    return info if isinstance(info, dict) else {}


def _episode_env_steps(episode: Any) -> int:
    value = getattr(episode, "env_steps", None)
    return int(value() if callable(value) else value or 0)


def _episode_agent_steps(episode: Any) -> int:
    value = getattr(episode, "agent_steps", None)
    return int(value() if callable(value) else value or 0)


def _remote_eval_worker_ids(eval_workers: Any) -> list[Any]:
    get_healthy_worker_ids = getattr(eval_workers, "healthy_worker_ids", None)
    if callable(get_healthy_worker_ids):
        return list(get_healthy_worker_ids())

    worker_manager = getattr(eval_workers, "_worker_manager", None)
    get_actor_ids = getattr(worker_manager, "actor_ids", None)
    if callable(get_actor_ids):
        return list(get_actor_ids())

    return []


def _foreach_one_eval_runner(eval_workers: Any, func) -> list[Any]:
    actor_ids = _remote_eval_worker_ids(eval_workers)
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


def _sample_one_eval_runner(eval_workers: Any) -> list[Any]:
    def sample_and_get_metrics(worker):
        return worker.sample(num_episodes=1), worker.get_metrics()

    return _foreach_one_eval_runner(eval_workers, sample_and_get_metrics)


def _next_evaluation_index(algorithm: Any) -> int:
    evaluation_index = int(getattr(algorithm, "_hardness_evaluation_index", 0)) + 1
    setattr(algorithm, "_hardness_evaluation_index", evaluation_index)
    return evaluation_index


def _training_iteration(algorithm: Any) -> int:
    return int(getattr(algorithm, "iteration", 0) or 0) + 1


def _success_rate_from_info(final_info: dict[str, Any]) -> float:
    success_rate = final_info.get("success_rate_650")
    if success_rate is not None:
        return float(success_rate)
    return 1.0 if bool(final_info.get("is_success", False)) else 0.0


def evaluate_and_save_hardness(
    algorithm: Any,
    eval_workers: Any,
    output_dir: str | Path | None = None,
) -> tuple[Any, int, int]:
    """RLlib custom evaluation hook that saves the sampled hardness artifact."""
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
    predicted_hardness = final_info.get("predicted_hardness")
    record = {
        "evaluation_index": _next_evaluation_index(algorithm),
        "training_iteration": _training_iteration(algorithm),
        "best_composition": dict(
            final_info.get("best_composition")
            or final_info.get("composition")
            or {}
        ),
        "predicted_hardness": predicted_hardness,
        "max_predicted_hardness": final_info.get(
            "max_predicted_hardness", predicted_hardness
        ),
        "uncertainty_hardness": final_info.get("uncertainty_hardness"),
        "success_rate_650": _success_rate_from_info(final_info),
    }
    save_hardness_evaluation(record, output_dir=output_dir)

    return eval_results, env_steps, agent_steps
