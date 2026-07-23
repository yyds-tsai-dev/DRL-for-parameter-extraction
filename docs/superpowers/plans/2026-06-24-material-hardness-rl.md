# Material Hardness RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the material hardness PPO path using a single-step hardness environment, bounded composition projection, shared `train_ppo.py` dispatch, hardness evaluation artifacts, W&B environment-specific project names, and TensorFlow as an optional extra.

**Architecture:** Keep PPO/Ray/Tune setup shared while moving EEHEMT and hardness wiring into environment-specific training modules. Add a focused `MaterialHardnessEnv` that turns raw policy actions into Feasible Material Compositions, calls the existing inference model, and emits hardness metrics. Keep projection, callbacks, evaluation, and training dispatch in separate files so each unit is directly testable.

**Tech Stack:** Python 3.11+, Gymnasium, NumPy, Pandas, Ray RLlib PPO/Tune, W&B, pytest, XGBoost model package via existing `env.InferenceModel`.

---

## File Structure

- Create `utils/composition_projection.py`: bounded simplex projection for material fraction vectors.
- Create `tests/test_composition_projection.py`: projection constraint and edge-case tests.
- Create `env/material_hardness_env.py`: Gymnasium single-step env for hardness optimization.
- Create `tests/test_material_hardness_env.py`: env reset/step/reward/fixed-feature tests with fake inference.
- Create `utils/hardness_callbacks.py`: RLlib callback for max hardness, best composition, uncertainty, success rate.
- Create `tests/test_hardness_callbacks.py`: callback metric logging tests with lightweight fakes.
- Create `evaluation/hardness_evaluation.py`: driver-side hardness evaluation CSV/JSON writer.
- Create `tests/test_hardness_evaluation.py`: artifact writing and metric return tests.
- Create `training/ppo_common.py`: shared PPO args, learner-resource setup, and W&B callback construction.
- Create `training/hardness_ppo.py`: hardness-specific parser args, env config, PPO config, checkpoint config, project name.
- Create `training/eehemt_ppo.py`: EEHEMT-specific parser args/config extracted from existing `train_ppo_tune.py`.
- Create `train_ppo.py`: shared entrypoint with `--env hardness` default and two-phase parser dispatch.
- Modify `tests/test_train_ppo_config.py`: import new training modules and verify dispatch/config behavior.
- Modify `scripts/train_ppo.sh`: call `train_ppo.py`; default hardness; allow caller args.
- Modify `pyproject.toml`: move `tensorflow` from core dependencies to `[project.optional-dependencies].keras-inference`.
- Delete `train_ppo_tune.py` after `train_ppo.py` replacement and test migration.

---

### Task 1: Bounded Composition Projection

**Files:**
- Create: `utils/composition_projection.py`
- Create: `tests/test_composition_projection.py`

- [ ] **Step 1: Write projection tests**

Create `tests/test_composition_projection.py`:

```python
import numpy as np

from utils.composition_projection import project_bounded_simplex


def test_project_bounded_simplex_preserves_feasible_vector():
    vector = np.array([0.2, 0.15, 0.15, 0.2, 0.15, 0.15], dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)

    assert np.allclose(projected, vector)
    assert np.isclose(projected.sum(), 1.0)


def test_project_bounded_simplex_enforces_bounds_and_sum():
    vector = np.array([2.0, -1.0, 0.3, 0.1, 0.8, -0.5], dtype=np.float64)

    projected = project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)

    assert projected.shape == (6,)
    assert np.all(projected >= 0.05 - 1e-8)
    assert np.all(projected <= 0.35 + 1e-8)
    assert np.isclose(projected.sum(), 1.0, atol=1e-8)


def test_project_bounded_simplex_rejects_infeasible_bounds():
    vector = np.zeros(6, dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.2, upper=0.35, target_sum=1.0)
    except ValueError as exc:
        assert "target_sum is outside feasible bounds" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_project_bounded_simplex_rejects_non_finite_values():
    vector = np.array([0.2, np.nan, 0.2, 0.2, 0.1, 0.1], dtype=np.float64)

    try:
        project_bounded_simplex(vector, lower=0.05, upper=0.35, target_sum=1.0)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
```

- [ ] **Step 2: Run projection tests to verify failure**

Run:

```bash
pytest tests/test_composition_projection.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.composition_projection'`.

- [ ] **Step 3: Implement bounded simplex projection**

Create `utils/composition_projection.py`:

```python
from __future__ import annotations

import numpy as np


def project_bounded_simplex(
    values,
    *,
    lower: float,
    upper: float,
    target_sum: float,
    atol: float = 1e-10,
) -> np.ndarray:
    """Project values to {x | lower <= x_i <= upper, sum(x) = target_sum}."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("values must be a one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("values must contain only finite numbers")
    if lower > upper:
        raise ValueError("lower must be less than or equal to upper")

    size = vector.size
    min_sum = lower * size
    max_sum = upper * size
    if target_sum < min_sum - atol or target_sum > max_sum + atol:
        raise ValueError(
            "target_sum is outside feasible bounds: "
            f"{target_sum} not in [{min_sum}, {max_sum}]"
        )

    theta_low = float(np.min(vector - upper))
    theta_high = float(np.max(vector - lower))
    for _ in range(100):
        theta = (theta_low + theta_high) / 2.0
        projected = np.clip(vector - theta, lower, upper)
        if projected.sum() > target_sum:
            theta_low = theta
        else:
            theta_high = theta

    projected = np.clip(vector - theta_high, lower, upper)
    correction = target_sum - float(projected.sum())
    if abs(correction) > atol:
        free = np.flatnonzero((projected > lower + atol) & (projected < upper - atol))
        if free.size:
            projected[free[0]] += correction
        else:
            index = int(np.argmax(projected)) if correction < 0 else int(np.argmin(projected))
            projected[index] = np.clip(projected[index] + correction, lower, upper)

    return projected.astype(np.float32)
```

- [ ] **Step 4: Run projection tests**

Run:

```bash
pytest tests/test_composition_projection.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit projection utility**

```bash
git add utils/composition_projection.py tests/test_composition_projection.py
git commit -m "feat: add bounded composition projection"
```

---

### Task 2: Material Hardness Environment

**Files:**
- Create: `env/material_hardness_env.py`
- Create: `tests/test_material_hardness_env.py`

- [ ] **Step 1: Write env tests with fake inference**

Create `tests/test_material_hardness_env.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from env.material_hardness_env import MaterialHardnessEnv


class FakeInferenceModel:
    def __init__(self, model_package_path):
        self.model_package_path = model_package_path
        self.inputs = []

    def predict(self, input_data, include_input=True):
        df = pd.DataFrame(input_data)
        self.inputs.append(df.copy())
        hardness = 650.0 + 100.0 * float(df.iloc[0]["frac_Ni"])
        return pd.DataFrame(
            {
                "Predicted hardness": [hardness],
                "Uncertainty hardness": [12.5],
            }
        )


def make_env():
    return MaterialHardnessEnv(
        {
            "model_package_path": "/tmp/fake.zip",
            "inference_model_cls": FakeInferenceModel,
            "hardness_threshold": 650.0,
            "reward_scale": 100.0,
            "reward_min": -3.0,
            "reward_max": 3.0,
        }
    )


def test_reset_returns_fixed_observation_inside_space():
    env = make_env()

    observation, info = env.reset(seed=123)

    assert observation.shape == (1,)
    assert env.observation_space.contains(observation)
    assert info == {}


def test_step_projects_action_injects_fixed_features_and_terminates():
    env = make_env()
    env.reset(seed=123)

    observation, reward, terminated, truncated, info = env.step(
        np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.25], dtype=np.float32)
    )

    composition = info["composition"]
    tunable_sum = sum(
        composition[name]
        for name in ("frac_Al", "frac_Cr", "frac_Mn", "frac_Fe", "frac_Co", "frac_Ni")
    )
    assert env.observation_space.contains(observation)
    assert terminated is True
    assert truncated is False
    assert np.isclose(tunable_sum, 1.0)
    assert all(0.05 <= composition[name] <= 0.35 for name in composition if name not in {"frac_Cu", "frac_Mo"})
    assert composition["frac_Cu"] == 0.0
    assert composition["frac_Mo"] == 0.0
    assert info["predicted_hardness"] >= 650.0
    assert info["uncertainty_hardness"] == 12.5
    assert info["is_success"] is True
    assert reward == np.clip(info["reward_unclipped"], -3.0, 3.0)


def test_step_handles_non_finite_prediction_as_failed_episode():
    class NonFiniteInferenceModel(FakeInferenceModel):
        def predict(self, input_data, include_input=True):
            return pd.DataFrame(
                {
                    "Predicted hardness": [np.nan],
                    "Uncertainty hardness": [1.0],
                }
            )

    env = MaterialHardnessEnv(
        {
            "model_package_path": "/tmp/fake.zip",
            "inference_model_cls": NonFiniteInferenceModel,
        }
    )
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))

    assert reward == -3.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is False
    assert "non-finite" in info["error"]
```

- [ ] **Step 2: Run env tests to verify failure**

Run:

```bash
pytest tests/test_material_hardness_env.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'env.material_hardness_env'`.

- [ ] **Step 3: Implement `MaterialHardnessEnv`**

Create `env/material_hardness_env.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from env import InferenceModel
from utils.composition_projection import project_bounded_simplex

TUNABLE_FRACTION_NAMES = (
    "frac_Al",
    "frac_Cr",
    "frac_Mn",
    "frac_Fe",
    "frac_Co",
    "frac_Ni",
)
FIXED_FRACTIONS = {"frac_Cu": 0.0, "frac_Mo": 0.0}


class MaterialHardnessEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        self.model_package_path = str(
            config.get(
                "model_package_path",
                os.getenv("HARDNESS_MODEL_PACKAGE_PATH", "env/hardness/XGB_model_selection_package.zip"),
            )
        )
        model_cls = config.get("inference_model_cls", InferenceModel)
        if model_cls is InferenceModel and not Path(self.model_package_path).exists():
            raise FileNotFoundError(f"Model package not found: {self.model_package_path}")
        self.model = model_cls(self.model_package_path)
        self.lower_fraction = float(config.get("lower_fraction", 0.05))
        self.upper_fraction = float(config.get("upper_fraction", 0.35))
        self.target_sum = float(config.get("target_sum", 1.0))
        self.hardness_threshold = float(config.get("hardness_threshold", 650.0))
        self.reward_scale = float(config.get("reward_scale", 100.0))
        self.reward_min = float(config.get("reward_min", -3.0))
        self.reward_max = float(config.get("reward_max", 3.0))

        self.action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.last_composition: dict[str, float] | None = None

    def _observation(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def _composition_from_action(self, action: np.ndarray) -> dict[str, float]:
        projected = project_bounded_simplex(
            action,
            lower=self.lower_fraction,
            upper=self.upper_fraction,
            target_sum=self.target_sum,
        )
        composition = {
            name: float(value)
            for name, value in zip(TUNABLE_FRACTION_NAMES, projected, strict=True)
        }
        composition.update(FIXED_FRACTIONS)
        return composition

    @staticmethod
    def _read_prediction(result_df: pd.DataFrame, column_prefix: str) -> float:
        matching_columns = [
            column for column in result_df.columns if column.lower() == column_prefix.lower()
        ]
        if not matching_columns:
            raise ValueError(f"Missing inference output column: {column_prefix}")
        return float(result_df.iloc[0][matching_columns[0]])

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.last_composition = None
        return self._observation(), {}

    def step(self, action):
        composition = self._composition_from_action(np.asarray(action, dtype=np.float32))
        self.last_composition = composition
        result_df = self.model.predict([composition], include_input=False)
        predicted_hardness = self._read_prediction(result_df, "Predicted hardness")
        uncertainty_hardness = self._read_prediction(result_df, "Uncertainty hardness")

        if not np.isfinite(predicted_hardness):
            return (
                self._observation(),
                self.reward_min,
                True,
                False,
                {
                    "composition": composition,
                    "predicted_hardness": predicted_hardness,
                    "uncertainty_hardness": uncertainty_hardness,
                    "reward_unclipped": self.reward_min,
                    "is_success": False,
                    "error": "non-finite predicted_hardness",
                },
            )

        reward_unclipped = (predicted_hardness - self.hardness_threshold) / self.reward_scale
        reward = float(np.clip(reward_unclipped, self.reward_min, self.reward_max))
        is_success = bool(predicted_hardness >= self.hardness_threshold)
        return (
            self._observation(),
            reward,
            True,
            False,
            {
                "composition": composition,
                "predicted_hardness": predicted_hardness,
                "uncertainty_hardness": uncertainty_hardness,
                "reward_unclipped": float(reward_unclipped),
                "is_success": is_success,
            },
        )
```

- [ ] **Step 4: Run env tests**

Run:

```bash
pytest tests/test_material_hardness_env.py tests/test_composition_projection.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit hardness env**

```bash
git add env/material_hardness_env.py tests/test_material_hardness_env.py
git commit -m "feat: add material hardness environment"
```

---

### Task 3: Hardness Metrics Callback

**Files:**
- Create: `utils/hardness_callbacks.py`
- Create: `tests/test_hardness_callbacks.py`

- [ ] **Step 1: Write callback tests**

Create `tests/test_hardness_callbacks.py`:

```python
from utils.hardness_callbacks import HardnessMetricsCallback


class FakeMetricsLogger:
    def __init__(self):
        self.values = {}

    def log_value(self, key, value, reduce=None):
        self.values[key] = (value, reduce)


class FakeEpisode:
    def __init__(self, infos):
        self.infos = infos


def test_hardness_callback_logs_max_hardness_and_success_rate():
    callback = HardnessMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        [
            {
                "predicted_hardness": 720.0,
                "uncertainty_hardness": 11.0,
                "composition": {"frac_Al": 0.2, "frac_Ni": 0.2},
                "is_success": True,
            }
        ]
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert metrics_logger.values["predicted_hardness"] == (720.0, "mean")
    assert metrics_logger.values["max_predicted_hardness"] == (720.0, "max")
    assert metrics_logger.values["uncertainty_hardness"] == (11.0, "mean")
    assert metrics_logger.values["success_rate_650"] == (1.0, "mean")
    assert callback.best_composition == {"frac_Al": 0.2, "frac_Ni": 0.2}


def test_hardness_callback_keeps_global_max():
    callback = HardnessMetricsCallback()
    metrics_logger = FakeMetricsLogger()

    for hardness in (700.0, 680.0):
        episode = FakeEpisode(
            [
                {
                    "predicted_hardness": hardness,
                    "uncertainty_hardness": 1.0,
                    "composition": {"frac_Ni": hardness},
                    "is_success": hardness >= 650.0,
                }
            ]
        )
        callback.on_episode_end(
            episode=episode,
            env_runner=None,
            metrics_logger=metrics_logger,
            env=None,
            env_index=0,
            rl_module=None,
        )

    assert metrics_logger.values["max_predicted_hardness"] == (700.0, "max")
    assert callback.best_composition == {"frac_Ni": 700.0}
```

- [ ] **Step 2: Run callback tests to verify failure**

Run:

```bash
pytest tests/test_hardness_callbacks.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.hardness_callbacks'`.

- [ ] **Step 3: Implement hardness callback**

Create `utils/hardness_callbacks.py`:

```python
from __future__ import annotations

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class HardnessMetricsCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.max_predicted_hardness = float("-inf")
        self.best_composition: dict[str, float] | None = None

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
        infos = getattr(episode, "infos", None) or []
        last_info = infos[-1] if infos else {}
        predicted_hardness = last_info.get("predicted_hardness")
        if predicted_hardness is None:
            return

        predicted_hardness = float(predicted_hardness)
        uncertainty_hardness = float(last_info.get("uncertainty_hardness", 0.0))
        is_success = bool(last_info.get("is_success", False))
        if predicted_hardness > self.max_predicted_hardness:
            self.max_predicted_hardness = predicted_hardness
            self.best_composition = dict(last_info.get("composition", {}))

        metrics_logger.log_value("predicted_hardness", predicted_hardness, reduce="mean")
        metrics_logger.log_value(
            "max_predicted_hardness",
            self.max_predicted_hardness,
            reduce="max",
        )
        metrics_logger.log_value(
            "uncertainty_hardness",
            uncertainty_hardness,
            reduce="mean",
        )
        metrics_logger.log_value(
            "success_rate_650",
            1.0 if is_success else 0.0,
            reduce="mean",
        )
```

- [ ] **Step 4: Run callback tests**

Run:

```bash
pytest tests/test_hardness_callbacks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit callback**

```bash
git add utils/hardness_callbacks.py tests/test_hardness_callbacks.py
git commit -m "feat: log hardness training metrics"
```

---

### Task 4: Hardness Evaluation Artifacts

**Files:**
- Create: `evaluation/hardness_evaluation.py`
- Create: `tests/test_hardness_evaluation.py`

- [ ] **Step 1: Write artifact writer tests**

Create `tests/test_hardness_evaluation.py`:

```python
import json
from types import SimpleNamespace

import pandas as pd

from evaluation.hardness_evaluation import evaluate_and_save_hardness, save_hardness_evaluation


def test_save_hardness_evaluation_writes_csv_and_json(tmp_path):
    record = {
        "evaluation_index": 1,
        "training_iteration": 2,
        "best_composition": {"frac_Al": 0.2, "frac_Ni": 0.2},
        "predicted_hardness": 720.0,
        "max_predicted_hardness": 720.0,
        "uncertainty_hardness": 11.0,
        "success_rate_650": 1.0,
    }

    paths = save_hardness_evaluation(record, output_dir=tmp_path)

    assert paths.csv_path.exists()
    assert paths.json_path.exists()
    csv_df = pd.read_csv(paths.csv_path)
    assert csv_df.iloc[0]["predicted_hardness"] == 720.0
    payload = json.loads(paths.json_path.read_text())
    assert payload["best_composition"]["frac_Al"] == 0.2


class FakeMetrics:
    def aggregate(self, metrics, key):
        self.metrics = metrics
        self.key = key

    def peek(self, key):
        return {"aggregated": True, "key": key}


class FakeAlgorithm:
    def __init__(self):
        self.metrics = FakeMetrics()
        self.iteration = 7


class FakeWorker:
    def sample(self, num_episodes):
        return [
            SimpleNamespace(
                infos=[
                    {
                        "composition": {"frac_Al": 0.2, "frac_Ni": 0.2},
                        "predicted_hardness": 720.0,
                        "uncertainty_hardness": 11.0,
                        "is_success": True,
                    }
                ],
                env_steps=1,
                agent_steps=1,
            )
        ]

    def get_metrics(self):
        return {"episodes": 1}


class FakeEvalWorkers:
    def foreach_env_runner(self, func, local_env_runner=True):
        return [func(FakeWorker())]


def test_evaluate_and_save_hardness_writes_artifact(tmp_path):
    eval_results, env_steps, agent_steps = evaluate_and_save_hardness(
        FakeAlgorithm(),
        FakeEvalWorkers(),
        output_dir=tmp_path,
    )

    assert eval_results["aggregated"] is True
    assert env_steps == 1
    assert agent_steps == 1
    assert list(tmp_path.glob("*.csv"))
    assert list(tmp_path.glob("*.json"))
```

- [ ] **Step 2: Run evaluation tests to verify failure**

Run:

```bash
pytest tests/test_hardness_evaluation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.hardness_evaluation'`.

- [ ] **Step 3: Implement evaluation artifact writer**

Create `evaluation/hardness_evaluation.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS


@dataclass(frozen=True)
class HardnessEvaluationPaths:
    csv_path: Path
    json_path: Path


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(record)
    composition = flattened.pop("best_composition", {}) or {}
    for key, value in composition.items():
        flattened[f"best_{key}"] = value
    return flattened


def save_hardness_evaluation(
    record: dict[str, Any],
    *,
    output_dir: str | Path,
) -> HardnessEvaluationPaths:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evaluation_index = int(record["evaluation_index"])
    training_iteration = int(record["training_iteration"])
    stem = f"eval_{evaluation_index:06d}_iter_{training_iteration:06d}"
    csv_path = output_path / f"{stem}.csv"
    json_path = output_path / f"{stem}.json"

    pd.DataFrame([_flatten_record(record)]).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return HardnessEvaluationPaths(csv_path=csv_path, json_path=json_path)


def _as_episode_list(sample_result: Any) -> list:
    if sample_result is None:
        return []
    if isinstance(sample_result, list):
        return sample_result
    if isinstance(sample_result, tuple):
        return list(sample_result)
    return [sample_result]


def _episode_final_info(episode: Any) -> dict[str, Any]:
    infos = getattr(episode, "infos", None) or []
    return infos[-1] if infos else {}


def _episode_env_steps(episode: Any) -> int:
    value = getattr(episode, "env_steps", None)
    return int(value() if callable(value) else value or 0)


def _episode_agent_steps(episode: Any) -> int:
    value = getattr(episode, "agent_steps", None)
    return int(value() if callable(value) else value or 0)


def _default_output_dir() -> Path:
    return Path("result") / "hardness-evaluation" / date.today().isoformat()


def _next_evaluation_index(algorithm: Any) -> int:
    evaluation_index = int(getattr(algorithm, "_hardness_evaluation_index", 0)) + 1
    setattr(algorithm, "_hardness_evaluation_index", evaluation_index)
    return evaluation_index


def evaluate_and_save_hardness(algorithm, eval_workers, output_dir: str | Path | None = None):
    def sample_and_get_metrics(worker):
        return worker.sample(num_episodes=1), worker.get_metrics()

    sample_and_metrics = eval_workers.foreach_env_runner(
        sample_and_get_metrics,
        local_env_runner=True,
    )
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

    predicted_hardness = float(final_info.get("predicted_hardness", float("nan")))
    record = {
        "evaluation_index": _next_evaluation_index(algorithm),
        "training_iteration": int(getattr(algorithm, "iteration", 0) or 0),
        "best_composition": final_info.get("composition", {}),
        "predicted_hardness": predicted_hardness,
        "max_predicted_hardness": predicted_hardness,
        "uncertainty_hardness": float(final_info.get("uncertainty_hardness", float("nan"))),
        "success_rate_650": 1.0 if bool(final_info.get("is_success", False)) else 0.0,
    }
    save_hardness_evaluation(record, output_dir=output_dir or _default_output_dir())
    return eval_results, env_steps, agent_steps
```

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
pytest tests/test_hardness_evaluation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit evaluation writer**

```bash
git add evaluation/hardness_evaluation.py tests/test_hardness_evaluation.py
git commit -m "feat: save hardness evaluation artifacts"
```

---

### Task 5: Shared PPO Common Module

**Files:**
- Create: `training/__init__.py`
- Create: `training/ppo_common.py`
- Create: `tests/test_ppo_common.py`

- [ ] **Step 1: Write common training tests**

Create `tests/test_ppo_common.py`:

```python
from types import SimpleNamespace

from training.ppo_common import (
    build_common_arg_parser,
    build_wandb_callback,
    resolve_learner_resources,
)


def test_common_parser_wandb_api_key_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "from-env")
    parser = build_common_arg_parser("/project")

    args = parser.parse_args([])

    assert args.wandb_api_key == "from-env"


def test_common_parser_wandb_api_key_cli_override(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "from-env")
    parser = build_common_arg_parser("/project")

    args = parser.parse_args(["--wandb_api_key", "from-cli"])

    assert args.wandb_api_key == "from-cli"


def test_build_wandb_callback_uses_env_project_name():
    args = SimpleNamespace(wandb_api_key="secret")

    callback = build_wandb_callback(args, project_name="project-name")

    assert callback.project == "project-name"
    assert callback.api_key == "secret"


def test_resolve_learner_resources_cpu(monkeypatch):
    monkeypatch.setattr("training.ppo_common.th.cuda.device_count", lambda: 0)
    monkeypatch.setenv("NUM_LEARNERS", "2")

    assert resolve_learner_resources() == (2, 0.0)
```

- [ ] **Step 2: Run common tests to verify failure**

Run:

```bash
pytest tests/test_ppo_common.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'training'`.

- [ ] **Step 3: Implement `training/ppo_common.py`**

Create `training/__init__.py`:

```python
"""Training configuration modules for PPO flows."""
```

Create `training/ppo_common.py`:

```python
from __future__ import annotations

import argparse
import os

import torch as th
from ray.air.integrations.wandb import WandbLoggerCallback


def build_common_arg_parser(current_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", choices=["hardness", "eehemt"], default="hardness")
    parser.add_argument(
        "--observation_filter",
        choices=["NoFilter", "MeanStdFilter"],
        default=os.getenv("OBSERVATION_FILTER", "NoFilter"),
    )
    parser.add_argument("--num_env_runners", type=int, default=int(os.getenv("NUM_ENV_RUNNERS", 4)))
    parser.add_argument(
        "--train_batch_size_per_learner",
        type=int,
        default=int(os.getenv("TRAIN_BATCH_SIZE_PER_LEARNER", 4096)),
    )
    parser.add_argument("--num_epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 5)))
    parser.add_argument("--minibatch_size", type=int, default=int(os.getenv("MINIBATCH_SIZE", 512)))
    parser.add_argument("--lr", type=float, default=float(os.getenv("LR", 5e-6)))
    parser.add_argument("--entropy_coeff", type=float, default=float(os.getenv("ENTROPY_COEFF", 5e-3)))
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--vf_loss_coeff", type=float, default=float(os.getenv("VF_LOSS_COEFF", 0.1)))
    parser.add_argument("--n_iterations", type=int, default=100)
    parser.add_argument("--restore_path", type=str, default=os.getenv("RESTORE_PATH", ""))
    parser.add_argument("--evaluation_interval", type=int, default=int(os.getenv("EVALUATION_INTERVAL", 2)))
    parser.add_argument(
        "--evaluation_num_env_runners",
        type=int,
        default=int(os.getenv("EVALUATION_NUM_ENV_RUNNERS", 1)),
    )
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join(current_dir, os.getenv("CHECKPOINT_DIR", "")))
    parser.add_argument("--wandb_api_key", type=str, default=os.getenv("WANDB_API_KEY", ""))
    return parser


def resolve_learner_resources() -> tuple[int, float]:
    device_count = th.cuda.device_count()
    if device_count > 0:
        num_learners = max(1, device_count // 2)
        return num_learners, 1.0
    return int(os.getenv("NUM_LEARNERS", 1)), 0.0


def build_wandb_callback(args, *, project_name: str) -> WandbLoggerCallback:
    return WandbLoggerCallback(project=project_name, api_key=args.wandb_api_key)
```

- [ ] **Step 4: Run common tests**

Run:

```bash
pytest tests/test_ppo_common.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit common training module**

```bash
git add training/__init__.py training/ppo_common.py tests/test_ppo_common.py
git commit -m "feat: add shared PPO training helpers"
```

---

### Task 6: Environment-Specific PPO Modules

**Files:**
- Create: `training/hardness_ppo.py`
- Create: `training/eehemt_ppo.py`
- Modify: `tests/test_train_ppo_config.py`

- [ ] **Step 1: Replace PPO config tests for environment modules**

Modify `tests/test_train_ppo_config.py` to import from the new modules and add hardness assertions:

```python
from types import SimpleNamespace

from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve
from training.eehemt_ppo import (
    EEHEMT_WANDB_PROJECT,
    build_checkpoint_config as build_eehemt_checkpoint_config,
    build_env_config as build_eehemt_env_config,
    build_ppo_config as build_eehemt_ppo_config,
)
from training.hardness_ppo import (
    HARDNESS_WANDB_PROJECT,
    build_checkpoint_config as build_hardness_checkpoint_config,
    build_env_config as build_hardness_env_config,
    build_ppo_config as build_hardness_ppo_config,
)
from evaluation.hardness_evaluation import evaluate_and_save_hardness
from utils.callbacks import TrainingMetricsCallback
from utils.hardness_callbacks import HardnessMetricsCallback


def test_eehemt_env_config_includes_ir_drop_parameters():
    args = SimpleNamespace(
        va_file_path="/tmp/model.va",
        csv_file_path="/tmp/data.csv",
        random_init=True,
        reduce_obs_err_dim=False,
        reward_norm=True,
        use_stagnation=False,
        rs_ext=1.25,
        rd_ext=0.4,
        ir_drop_n_iter=3,
        ir_drop_maxfev=123,
        nrmse_threshold=7.5,
    )

    assert build_eehemt_env_config(args)["ir_drop_maxfev"] == 123


def test_eehemt_checkpoint_config_ranks_by_lowest_nrmse():
    checkpoint_config = build_eehemt_checkpoint_config()

    assert checkpoint_config.checkpoint_score_attribute == "env_runners/min_nrmse"
    assert checkpoint_config.checkpoint_score_order == "min"


def test_hardness_env_config_uses_model_package_and_reward_settings():
    args = SimpleNamespace(
        hardness_model_package_path="env/hardness/XGB_model_selection_package.zip",
        hardness_threshold=650.0,
        hardness_reward_scale=100.0,
        hardness_reward_min=-3.0,
        hardness_reward_max=3.0,
    )

    env_config = build_hardness_env_config(args)

    assert env_config["model_package_path"] == "env/hardness/XGB_model_selection_package.zip"
    assert env_config["hardness_threshold"] == 650.0
    assert env_config["reward_min"] == -3.0
    assert env_config["reward_max"] == 3.0


def test_hardness_checkpoint_config_ranks_by_max_predicted_hardness():
    checkpoint_config = build_hardness_checkpoint_config()

    assert checkpoint_config.checkpoint_score_attribute == "env_runners/max_predicted_hardness"
    assert checkpoint_config.checkpoint_score_order == "max"


def test_eehemt_ppo_config_wires_callbacks_and_evaluation():
    args = SimpleNamespace(
        va_file_path="/tmp/model.va",
        csv_file_path="/tmp/data.csv",
        random_init=True,
        reduce_obs_err_dim=False,
        reward_norm=True,
        use_stagnation=False,
        rs_ext=1.25,
        rd_ext=0.4,
        ir_drop_n_iter=3,
        ir_drop_maxfev=123,
        nrmse_threshold=7.5,
        num_env_runners=2,
        observation_filter="NoFilter",
        train_batch_size_per_learner=128,
        num_epochs=2,
        minibatch_size=64,
        lr=1e-5,
        entropy_coeff=0.01,
        grad_clip=1.0,
        vf_loss_coeff=0.1,
        evaluation_interval=3,
        evaluation_num_env_runners=1,
    )

    config = build_eehemt_ppo_config(args, num_learners=1, num_gpus_per_learner=0.0)

    assert config.callbacks_class is TrainingMetricsCallback
    assert config.custom_evaluation_function is evaluate_and_plot_iv_curve


def test_hardness_ppo_config_wires_hardness_callback():
    args = SimpleNamespace(
        hardness_model_package_path="env/hardness/XGB_model_selection_package.zip",
        hardness_threshold=650.0,
        hardness_reward_scale=100.0,
        hardness_reward_min=-3.0,
        hardness_reward_max=3.0,
        num_env_runners=2,
        observation_filter="NoFilter",
        train_batch_size_per_learner=128,
        num_epochs=2,
        minibatch_size=64,
        lr=1e-5,
        entropy_coeff=0.01,
        grad_clip=1.0,
        vf_loss_coeff=0.1,
        evaluation_interval=3,
        evaluation_num_env_runners=1,
    )

    config = build_hardness_ppo_config(args, num_learners=1, num_gpus_per_learner=0.0)

    assert config.callbacks_class is HardnessMetricsCallback
    assert config.custom_evaluation_function is evaluate_and_save_hardness
    assert config.evaluation_interval == 3


def test_wandb_project_names_are_environment_specific():
    assert HARDNESS_WANDB_PROJECT == "PPO_for_material_hardness_optimization"
    assert EEHEMT_WANDB_PROJECT == "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"
```

- [ ] **Step 2: Run PPO config tests to verify failure**

Run:

```bash
pytest tests/test_train_ppo_config.py -q
```

Expected: FAIL with missing `training.eehemt_ppo` or `training.hardness_ppo`.

- [ ] **Step 3: Implement `training/eehemt_ppo.py`**

Create `training/eehemt_ppo.py` by moving EEHEMT-specific functions from `train_ppo_tune.py`:

```python
from __future__ import annotations

import argparse
import os

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from env.eehemt_env import EEHEMTEnv_Measure_VDS
from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve
from utils.callbacks import TrainingMetricsCallback

EEHEMT_WANDB_PROJECT = "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"


def add_env_args(parser: argparse.ArgumentParser, current_dir: str) -> None:
    parser.add_argument("--va_file_path", type=str, default=os.path.join(current_dir, os.getenv("VA_FILE_PATH", "")))
    parser.add_argument("--csv_file_path", type=str, default=os.path.join(current_dir, os.getenv("CSV_FILE_PATH", "")))
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--reduce_obs_err_dim", action="store_true")
    parser.add_argument("--reward_norm", action="store_true")
    parser.add_argument("--use_stagnation", action="store_true")
    parser.add_argument("--rs_ext", type=float, default=float(os.getenv("RS_EXT", 0.0)))
    parser.add_argument("--rd_ext", type=float, default=float(os.getenv("RD_EXT", 0.0)))
    parser.add_argument("--ir_drop_n_iter", type=int, default=int(os.getenv("IR_DROP_N_ITER", 2)))
    parser.add_argument("--ir_drop_maxfev", type=int, default=int(os.getenv("IR_DROP_MAXFEV", 40)))
    parser.add_argument("--nrmse_threshold", type=float, default=float(os.getenv("NRMSE_THRESHOLD", 10.0)))


def build_env_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "va_file_path": args.va_file_path,
        "csv_file_path": args.csv_file_path,
        "random_init": args.random_init,
        "reduce_obs_err_dim": args.reduce_obs_err_dim,
        "reward_norm": args.reward_norm,
        "use_stagnation": args.use_stagnation,
        "rs_ext": args.rs_ext,
        "rd_ext": args.rd_ext,
        "ir_drop_n_iter": args.ir_drop_n_iter,
        "ir_drop_maxfev": args.ir_drop_maxfev,
        "nrmse_threshold": args.nrmse_threshold,
    }


def build_ppo_config(args, *, num_learners: int, num_gpus_per_learner: float) -> PPOConfig:
    return (
        PPOConfig()
        .environment(env=EEHEMTEnv_Measure_VDS, env_config=build_env_config(args))
        .env_runners(num_env_runners=args.num_env_runners, observation_filter=args.observation_filter)
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr * num_learners,
            entropy_coeff=args.entropy_coeff,
            grad_clip=args.grad_clip,
            vf_loss_coeff=args.vf_loss_coeff,
            vf_clip_param=20.0,
        )
        .learners(num_learners=num_learners, num_gpus_per_learner=num_gpus_per_learner)
        .callbacks(callbacks_class=TrainingMetricsCallback)
        .evaluation(
            evaluation_interval=args.evaluation_interval,
            evaluation_num_env_runners=args.evaluation_num_env_runners,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            custom_evaluation_function=evaluate_and_plot_iv_curve,
            evaluation_config={"explore": False},
        )
    )


def build_checkpoint_config() -> tune.CheckpointConfig:
    return tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="env_runners/min_nrmse",
        checkpoint_score_order="min",
    )
```

- [ ] **Step 4: Implement `training/hardness_ppo.py`**

Create `training/hardness_ppo.py`:

```python
from __future__ import annotations

import argparse
import os

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from env.material_hardness_env import MaterialHardnessEnv
from evaluation.hardness_evaluation import evaluate_and_save_hardness
from utils.hardness_callbacks import HardnessMetricsCallback

HARDNESS_WANDB_PROJECT = "PPO_for_material_hardness_optimization"


def add_env_args(parser: argparse.ArgumentParser, current_dir: str) -> None:
    parser.add_argument(
        "--hardness_model_package_path",
        type=str,
        default=os.path.join(
            current_dir,
            os.getenv("HARDNESS_MODEL_PACKAGE_PATH", "env/hardness/XGB_model_selection_package.zip"),
        ),
    )
    parser.add_argument("--hardness_threshold", type=float, default=float(os.getenv("HARDNESS_THRESHOLD", 650.0)))
    parser.add_argument("--hardness_reward_scale", type=float, default=float(os.getenv("HARDNESS_REWARD_SCALE", 100.0)))
    parser.add_argument("--hardness_reward_min", type=float, default=float(os.getenv("HARDNESS_REWARD_MIN", -3.0)))
    parser.add_argument("--hardness_reward_max", type=float, default=float(os.getenv("HARDNESS_REWARD_MAX", 3.0)))


def build_env_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_package_path": args.hardness_model_package_path,
        "hardness_threshold": args.hardness_threshold,
        "reward_scale": args.hardness_reward_scale,
        "reward_min": args.hardness_reward_min,
        "reward_max": args.hardness_reward_max,
    }


def build_ppo_config(args, *, num_learners: int, num_gpus_per_learner: float) -> PPOConfig:
    return (
        PPOConfig()
        .environment(env=MaterialHardnessEnv, env_config=build_env_config(args))
        .env_runners(num_env_runners=args.num_env_runners, observation_filter=args.observation_filter)
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr * num_learners,
            entropy_coeff=args.entropy_coeff,
            grad_clip=args.grad_clip,
            vf_loss_coeff=args.vf_loss_coeff,
            vf_clip_param=20.0,
        )
        .learners(num_learners=num_learners, num_gpus_per_learner=num_gpus_per_learner)
        .callbacks(callbacks_class=HardnessMetricsCallback)
        .evaluation(
            evaluation_interval=args.evaluation_interval,
            evaluation_num_env_runners=args.evaluation_num_env_runners,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            custom_evaluation_function=evaluate_and_save_hardness,
            evaluation_config={"explore": False},
        )
    )


def build_checkpoint_config() -> tune.CheckpointConfig:
    return tune.CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="env_runners/max_predicted_hardness",
        checkpoint_score_order="max",
    )
```

- [ ] **Step 5: Run PPO config tests**

Run:

```bash
pytest tests/test_train_ppo_config.py tests/test_ppo_common.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit environment-specific training modules**

```bash
git add training/eehemt_ppo.py training/hardness_ppo.py tests/test_train_ppo_config.py
git commit -m "feat: split PPO config by environment"
```

---

### Task 7: Shared Entrypoint Rename And Dispatch

**Files:**
- Create: `train_ppo.py`
- Delete: `train_ppo_tune.py`
- Modify: `tests/test_train_ppo_config.py`

- [ ] **Step 1: Add dispatch parser tests**

Append to `tests/test_train_ppo_config.py`:

```python
from train_ppo import build_arg_parser, select_training_module


def test_train_ppo_defaults_to_hardness_env():
    parser = build_arg_parser("/project")

    args = parser.parse_args([])

    assert args.env == "hardness"


def test_train_ppo_can_build_eehemt_parser_from_argv():
    parser = build_arg_parser("/project", ["--env", "eehemt"])

    args = parser.parse_args(["--env", "eehemt", "--va_file_path", "/tmp/model.va"])

    assert args.env == "eehemt"
    assert args.va_file_path == "/tmp/model.va"


def test_select_training_module_dispatches_by_env():
    hardness_module = select_training_module("hardness")
    eehemt_module = select_training_module("eehemt")

    assert hardness_module.HARDNESS_WANDB_PROJECT == "PPO_for_material_hardness_optimization"
    assert eehemt_module.EEHEMT_WANDB_PROJECT == "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"
```

- [ ] **Step 2: Run dispatch tests to verify failure**

Run:

```bash
pytest tests/test_train_ppo_config.py::test_train_ppo_defaults_to_hardness_env tests/test_train_ppo_config.py::test_select_training_module_dispatches_by_env -q
```

Expected: FAIL because `train_ppo.py` does not exist.

- [ ] **Step 3: Implement `train_ppo.py`**

Create `train_ppo.py`:

```python
from __future__ import annotations

import argparse
import os
import sys

import ray
from dotenv import load_dotenv
from ray import tune
from ray.rllib.algorithms.ppo import PPO

from training.ppo_common import (
    build_common_arg_parser,
    build_wandb_callback,
    resolve_learner_resources,
)
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def select_training_module(env_name: str):
    if env_name == "hardness":
        from training import hardness_ppo

        return hardness_ppo
    if env_name == "eehemt":
        from training import eehemt_ppo

        return eehemt_ppo
    raise ValueError(f"Unsupported env: {env_name}")


def build_arg_parser(
    current_dir: str,
    argv: list[str] | None = None,
) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env", choices=["hardness", "eehemt"], default="hardness")
    pre_args, _ = pre_parser.parse_known_args([] if argv is None else argv)
    training_module = select_training_module(pre_args.env)

    parser = argparse.ArgumentParser(parents=[build_common_arg_parser(current_dir)])
    training_module.add_env_args(parser, current_dir)
    return parser


def build_ray_runtime_env(current_dir: str) -> dict[str, object]:
    return {
        "env_vars": {"PROJECT_ROOT": current_dir},
        "excludes": [
            ".git/**",
            ".venv/**",
            ".mypy_cache/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            ".codebase-memory/**",
            "result/**",
            "demo/demo.tar.gz",
            "**/__pycache__/**",
            "**/*.pyc",
        ],
    }


def main() -> None:
    load_dotenv()
    configure_logging()
    current_dir = os.getcwd()
    os.environ.setdefault("PROJECT_ROOT", current_dir)
    parser = build_arg_parser(current_dir, sys.argv[1:])
    args = parser.parse_args()
    training_module = select_training_module(args.env)
    num_learners, num_gpus_per_learner = resolve_learner_resources()
    config = training_module.build_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
    )

    ray.init(ignore_reinit_error=True, runtime_env=build_ray_runtime_env(current_dir))
    if args.restore_path:
        logger.info("Restoring training from: %s", args.restore_path)
        tuner = tune.Tuner.restore(
            path=args.restore_path,
            trainable=PPO,
            resume_unfinished=True,
            resume_errored=True,
            param_space=config,
        )
    else:
        run_config = tune.RunConfig(
            name=f"{args.env.upper()}_PPO",
            storage_path=args.checkpoint_dir,
            stop={"training_iteration": args.n_iterations},
            checkpoint_config=training_module.build_checkpoint_config(),
            callbacks=[
                build_wandb_callback(
                    args,
                    project_name=(
                        training_module.HARDNESS_WANDB_PROJECT
                        if args.env == "hardness"
                        else training_module.EEHEMT_WANDB_PROJECT
                    ),
                )
            ],
        )
        tuner = tune.Tuner(PPO, param_space=config, run_config=run_config)

    results = tuner.fit()
    if not args.restore_path:
        completed_iterations = [
            result.metrics.get("training_iteration", 0)
            for result in results
            if result.metrics
        ]
        if not completed_iterations or max(completed_iterations) < args.n_iterations:
            raise SystemExit(130)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Delete old entrypoint**

Run:

```bash
git rm train_ppo_tune.py
```

Expected: `rm 'train_ppo_tune.py'`.

- [ ] **Step 5: Run dispatch and config tests**

Run:

```bash
pytest tests/test_train_ppo_config.py tests/test_ppo_common.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit entrypoint rename**

```bash
git add train_ppo.py tests/test_train_ppo_config.py
git commit -m "feat: dispatch PPO training by environment"
```

---

### Task 8: Script And Dependency Migration

**Files:**
- Modify: `scripts/train_ppo.sh`
- Modify: `pyproject.toml`
- Modify: `tests/test_train_ppo_config.py`
- Modify: `tests/test_model_inference_package.py`

- [ ] **Step 1: Add script and dependency tests**

Append to `tests/test_train_ppo_config.py`:

```python
def test_train_ppo_script_calls_new_entrypoint():
    with open("scripts/train_ppo.sh", encoding="utf-8") as script_file:
        script = script_file.read()

    assert "python train_ppo.py" in script
    assert "train_ppo_tune.py" not in script
```

Append to `tests/test_model_inference_package.py`:

```python
def test_tensorflow_is_optional_core_dependency():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    dependencies_block = pyproject.split("[dependency-groups]", maxsplit=1)[0]
    assert '"tensorflow' not in dependencies_block
    assert "keras-inference" in pyproject
    assert '"tensorflow>=2.19.0"' in pyproject
```

- [ ] **Step 2: Run migration tests to verify failure**

Run:

```bash
pytest tests/test_train_ppo_config.py::test_train_ppo_script_calls_new_entrypoint tests/test_model_inference_package.py::test_tensorflow_is_optional_core_dependency -q
```

Expected: FAIL because script still calls `train_ppo_tune.py` and TensorFlow is still core dependency.

- [ ] **Step 3: Update script**

Replace `scripts/train_ppo.sh` with:

```bash
#!/bin/bash

set -euo pipefail

source .venv/bin/activate
python train_ppo.py "$@"
```

- [ ] **Step 4: Move TensorFlow to optional extra**

Edit `pyproject.toml` so the dependency section removes:

```toml
"tensorflow>=2.19.0",
```

Add before `[dependency-groups]`:

```toml
[project.optional-dependencies]
keras-inference = [
    "tensorflow>=2.19.0",
]
```

- [ ] **Step 5: Run migration tests**

Run:

```bash
pytest tests/test_train_ppo_config.py::test_train_ppo_script_calls_new_entrypoint tests/test_model_inference_package.py::test_tensorflow_is_optional_core_dependency -q
```

Expected: PASS.

- [ ] **Step 6: Run uv lock refresh if available**

Run:

```bash
uv lock
```

Expected: exits 0 and updates `uv.lock` if dependency metadata changed.

- [ ] **Step 7: Commit script and dependency changes**

```bash
git add scripts/train_ppo.sh pyproject.toml uv.lock tests/test_train_ppo_config.py tests/test_model_inference_package.py
git commit -m "chore: update PPO entrypoint and optional tensorflow extra"
```

---

### Task 9: Full Verification Sweep

**Files:**
- No planned source changes unless tests reveal defects.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest \
  tests/test_composition_projection.py \
  tests/test_material_hardness_env.py \
  tests/test_hardness_callbacks.py \
  tests/test_hardness_evaluation.py \
  tests/test_ppo_common.py \
  tests/test_train_ppo_config.py \
  tests/test_model_inference_package.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run existing relevant regression tests**

Run:

```bash
pytest tests/test_env_measure_vds.py tests/test_iv_curve_evaluation.py tests/test_callbacks.py -q
```

Expected: PASS.

- [ ] **Step 3: Run lint on changed files**

Run:

```bash
ruff check \
  train_ppo.py \
  training \
  env/material_hardness_env.py \
  utils/composition_projection.py \
  utils/hardness_callbacks.py \
  evaluation/hardness_evaluation.py \
  tests/test_composition_projection.py \
  tests/test_material_hardness_env.py \
  tests/test_hardness_callbacks.py \
  tests/test_hardness_evaluation.py \
  tests/test_ppo_common.py \
  tests/test_train_ppo_config.py \
  tests/test_model_inference_package.py
```

Expected: PASS.

- [ ] **Step 4: Run CLI smoke tests**

Run:

```bash
python train_ppo.py --help
python train_ppo.py --env eehemt --help
python scripts/run_model_inference.py --help
```

Expected: each exits 0 and prints usage/help.

- [ ] **Step 5: Confirm no uncommitted verification changes remain**

Run:

```bash
git status --short
```

Expected: no output. If output appears, inspect each file and either commit the intentional fix with a specific message or revert only the accidental file generated by verification.
