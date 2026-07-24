# P3: Shared Eval Plumbing + Toy-Problem Acceptance + User Docs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deduplicate the RLlib evaluation plumbing that both custom eval functions copy, add the threshold-agnostic `success_rate` metric, prove the extension model with an in-test toy problem registered with zero shared-file edits, and document how a future user adds a problem.

**Architecture:** New `evaluation/rllib_plumbing.py` holds the six helpers that are byte-identical between `evaluation/hardness_evaluation.py` and `evaluation/iv_curve_evaluation.py`; both files re-import them under their private names so tests and callers are untouched. `_episode_final_info` deliberately stays per-file — the two variants have different exception/fallback semantics and unifying them would change behavior. `utils/hardness_callbacks.py` additionally emits `success_rate` (threshold-agnostic twin of the legacy `success_rate_650`, which stays for dashboard continuity per ADR 0003). A new test registers a complete toy problem from test code only — the executable proof that `--env <new>` needs no shared-code edits. `docs/how-to-add-a-problem.md` and three new CONTEXT.md terms make the extension path discoverable.

**Tech Stack:** Python 3.11, uv, pytest, Ray RLlib, gymnasium.

**Design spec:** `docs/superpowers/specs/2026-07-23-pluggable-problem-architecture-design.md` (sections 4.5, 4.6). Controller refinement recorded here: only byte-identical helpers are extracted; `_episode_final_info` variants stay per-file (differing semantics), and the callback/eval files keep their public names and metric strings verbatim.

## Global Constraints

- Emitted metric names unchanged: `predicted_hardness`, `max_predicted_hardness`, `uncertainty_hardness`, `success_rate_650` all keep being logged exactly as today; `success_rate` is ADDED, never a rename.
- Artifact schema of `evaluate_and_save_hardness` records unchanged (`success_rate_650` key stays in the saved record).
- All existing tests stay green UNMODIFIED. Verification trio after every task: `uv run pytest`, `uv run ruff check .`, `uv run mypy .` exit 0 (always via `uv run` — the global mypy lacks the venv packages and false-alarms).
- Every commit message ends with the trailer line: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Work on branch `pluggable-problem-architecture`. Never commit `.claude/` or `CLAUDE.md`.
- Test counts: start `129 passed`; Task 1 +0 (pure extraction), Task 2 +1 (130), Task 3 +3 (133), Task 4 +0, Task 5 wrap-up expects `133 passed`.

---

### Task 1: Extract byte-identical eval plumbing into `evaluation/rllib_plumbing.py`

**Files:**
- Create: `evaluation/rllib_plumbing.py`
- Modify: `evaluation/hardness_evaluation.py`, `evaluation/iv_curve_evaluation.py` (replace local defs with facade imports)
- Test (must stay green UNMODIFIED): `tests/test_hardness_evaluation.py`, `tests/test_iv_curve_evaluation.py`, `tests/test_plot_evaluation_curves.py`

**Interfaces:**
- Produces: `evaluation.rllib_plumbing` exporting `as_episode_list`, `episode_env_steps`, `episode_agent_steps`, `remote_eval_worker_ids`, `foreach_one_eval_runner`, `sample_one_eval_runner`.
- NOT extracted (stays per-file, differing semantics): `_episode_final_info`, `_next_evaluation_index` (different attr names), output-dir helpers, artifact writers, plot collectors.

- [x] **Step 1: Create `evaluation/rllib_plumbing.py`**

```python
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
```

- [x] **Step 2: Facade the helpers in `evaluation/hardness_evaluation.py`**

Delete the six local definitions (`_as_episode_list`, `_episode_env_steps`, `_episode_agent_steps`, `_remote_eval_worker_ids`, `_foreach_one_eval_runner`, `_sample_one_eval_runner`) and add, after the existing `from ray.rllib.utils.metrics import ...` import:

```python
from evaluation.rllib_plumbing import (
    as_episode_list as _as_episode_list,
    episode_agent_steps as _episode_agent_steps,
    episode_env_steps as _episode_env_steps,
    foreach_one_eval_runner as _foreach_one_eval_runner,
    remote_eval_worker_ids as _remote_eval_worker_ids,
    sample_one_eval_runner as _sample_one_eval_runner,
)
```

`_episode_final_info`, `_next_evaluation_index`, `_training_iteration`, `_success_rate_from_info`, the artifact writer, and `evaluate_and_save_hardness` stay untouched. (`_foreach_one_eval_runner` and `_remote_eval_worker_ids` keep working for any test that imports them by their private names.)

- [x] **Step 3: Facade the helpers in `evaluation/iv_curve_evaluation.py`**

Delete the same six local definitions there and add the identical import block (after `from ray.rllib.utils.metrics import ...`). `_episode_final_info`, `_unwrap_env`, `_static_plot_data_from_env_runner`, `_collect_static_plot_data`, `_plot_dir`, `_next_evaluation_index`, and `evaluate_and_plot_iv_curve` stay untouched. NOTE: the iv variant of `_foreach_one_eval_runner` called `eval_workers.foreach_env_runner` directly while the extracted version uses `getattr(...)` first — semantically identical (the attribute is always present on real and fake worker groups; the hardness variant already shipped the getattr form).

- [x] **Step 4: Run the three eval test files**

Run: `uv run pytest tests/test_hardness_evaluation.py tests/test_iv_curve_evaluation.py tests/test_plot_evaluation_curves.py -v`
Expected: ALL PASS unmodified. If any test imports a deleted private name and fails at collection, the facade import block above restores that exact name — re-check spelling rather than editing tests. If a test asserts semantics the extraction changed, STOP and report BLOCKED with the failing output.

- [x] **Step 5: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `129 passed`; ruff clean; mypy exit 0.

```bash
git add evaluation/rllib_plumbing.py evaluation/hardness_evaluation.py evaluation/iv_curve_evaluation.py
git commit -m "refactor: extract shared RLlib eval plumbing helpers" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Threshold-agnostic `success_rate` metric (TDD)

**Files:**
- Modify: `utils/hardness_callbacks.py`
- Modify: `tests/test_hardness_callbacks.py` (append one test; existing tests untouched)

**Interfaces:**
- Produces: `HardnessMetricsCallback` logs `success_rate` (mean-reduced 0/1) in addition to the legacy `success_rate_650`.

- [x] **Step 1: Append the failing test to `tests/test_hardness_callbacks.py`**

The file already defines `FakeMetricsLogger` (records `values[key] = (value, reduce)`) and `FakeEpisode(infos)` — reuse them. Append exactly:

```python
def test_hardness_callback_logs_threshold_agnostic_success_rate():
    callback = HardnessMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        [
            {
                "predicted_hardness": 700.0,
                "uncertainty_hardness": 2.0,
                "composition": {"frac_Ni": 0.2},
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

    assert metrics_logger.values["success_rate"] == (1.0, "mean")
    assert metrics_logger.values["success_rate_650"] == (1.0, "mean")
```

- [x] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_hardness_callbacks.py -v`
Expected: the new test FAILS (`success_rate` never logged); the existing 2 tests still pass.

- [x] **Step 3: Add the metric in `utils/hardness_callbacks.py`**

Directly after the existing `success_rate_650` log call, add:

```python
        metrics_logger.log_value(
            "success_rate", 1.0 if is_success else 0.0, reduce="mean"
        )
```

(`success_rate_650` stays — dashboard continuity per ADR 0003.)

- [x] **Step 4: Run to verify green, full trio, commit**

Run: `uv run pytest tests/test_hardness_callbacks.py -v` → all pass.
Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `130 passed`; ruff clean; mypy exit 0.

```bash
git add utils/hardness_callbacks.py tests/test_hardness_callbacks.py
git commit -m "feat: log threshold-agnostic success_rate alongside legacy metric" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Toy-problem acceptance test — zero shared-file edits (TDD-by-construction)

**Files:**
- Create: `tests/test_toy_problem_extension.py`
- NO other file may change — that is the point of the test.

**Interfaces:**
- Consumes: `problems.registry` (P1), `training.ppo_common.build_base_ppo_config` (P1), `env.backends.PredictionResult` (P2), `env.objectives.ThresholdMaximizeObjective` (P2), `train_ppo.build_arg_parser`/`select_training_module`.

- [x] **Step 1: Create `tests/test_toy_problem_extension.py`**

```python
"""Acceptance proof for the pluggable-problem architecture (ADR 0003).

A complete third problem — env, backend, objective, training assembly — is
assembled and registered HERE, in test code only. If these tests pass without
any edit to train_ppo.py, training/, problems/, env/, or evaluation/, the
"adapter + registration, zero shared-file edits" contract holds.
"""

from __future__ import annotations

import argparse
import types

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box

import problems  # noqa: F401  (registers builtins)
import train_ppo
from env.backends import PredictionResult
from env.objectives import ThresholdMaximizeObjective
from problems import registry
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from training.ppo_common import build_base_ppo_config


class ToyBackend:
    """Deterministic stand-in for an ANN surrogate."""

    def __init__(self, model_package_path):
        self.model_package_path = model_package_path

    def predict(self, features):
        strength = 100.0 * float(sum(features.values()))
        return PredictionResult(
            values={"strength": strength}, uncertainties={"strength": 1.0}
        )

    def close(self):
        pass


class ToyStrengthEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}
        self.backend = ToyBackend(config.get("model_package_path", "unused"))
        self.objective = ThresholdMaximizeObjective(
            threshold=float(config.get("strength_threshold", 50.0)),
            scale=10.0,
            reward_min=-1.0,
            reward_max=1.0,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        features = {"a": float(action[0]), "b": float(action[1])}
        result = self.backend.predict(features)
        outcome = self.objective.evaluate(result.values["strength"])
        info = {"predicted_strength": result.values["strength"]}
        return np.zeros(1, dtype=np.float32), outcome.reward, True, False, info


def _toy_add_env_args(parser, current_dir):
    parser.add_argument("--strength_threshold", type=float, default=50.0)
    return parser


def _toy_build_env_config(args):
    return {"strength_threshold": args.strength_threshold}


def _toy_build_ppo_config(args, *, num_learners, num_gpus_per_learner):
    return build_base_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        env_cls=ToyStrengthEnv,
        env_config=_toy_build_env_config(args),
        callbacks_class=DefaultCallbacks,
        custom_evaluation_function=None,
    )


def _toy_build_checkpoint_config():
    return types.SimpleNamespace(
        checkpoint_score_attribute=ThresholdMaximizeObjective.RANKED_METRIC,
        checkpoint_score_order=ThresholdMaximizeObjective.RANKED_ORDER,
    )


def _build_toy_module():
    module = types.ModuleType("toy_strength_problem")
    module.TOY_WANDB_PROJECT = "PPO_for_toy_strength"
    module.add_env_args = _toy_add_env_args
    module.build_env_config = _toy_build_env_config
    module.build_ppo_config = _toy_build_ppo_config
    module.build_checkpoint_config = _toy_build_checkpoint_config
    return module


def _toy_spec():
    module = _build_toy_module()
    return registry.ProblemSpec(
        name="toy_strength",
        module=module,
        wandb_project=module.TOY_WANDB_PROJECT,
        checkpoint_metric=ThresholdMaximizeObjective.RANKED_METRIC,
        checkpoint_order=ThresholdMaximizeObjective.RANKED_ORDER,
        add_env_args=module.add_env_args,
        build_env_config=module.build_env_config,
        build_ppo_config=module.build_ppo_config,
        build_checkpoint_config=module.build_checkpoint_config,
    )


@pytest.fixture()
def toy_registered():
    saved = registry.snapshot()
    registry.register(_toy_spec())
    yield
    registry.restore(saved)


def test_toy_problem_reaches_cli_and_dispatch_without_shared_edits(toy_registered):
    parser = train_ppo.build_arg_parser("/project", ["--env", "toy_strength"])
    args = parser.parse_args(["--env", "toy_strength", "--strength_threshold", "75.0"])

    assert "toy_strength" in registry.names()
    assert args.env == "toy_strength"
    assert args.strength_threshold == 75.0
    assert (
        train_ppo.select_training_module("toy_strength").TOY_WANDB_PROJECT
        == "PPO_for_toy_strength"
    )


def test_toy_problem_builds_ppo_config_through_generic_assembly(toy_registered):
    from types import SimpleNamespace

    args = SimpleNamespace(
        num_env_runners=1,
        observation_filter="NoFilter",
        train_batch_size_per_learner=64,
        num_epochs=1,
        minibatch_size=32,
        lr=1e-5,
        entropy_coeff=0.0,
        grad_clip=1.0,
        vf_loss_coeff=0.1,
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        strength_threshold=50.0,
    )

    spec = registry.get("toy_strength")
    config = spec.build_ppo_config(args, num_learners=1, num_gpus_per_learner=0.0)

    assert config.env is ToyStrengthEnv
    assert config.observation_filter == "NoFilter"
    checkpoint = spec.build_checkpoint_config()
    assert (
        checkpoint.checkpoint_score_attribute
        == ThresholdMaximizeObjective.RANKED_METRIC
    )


def test_toy_env_episode_uses_backend_and_objective(toy_registered):
    env = ToyStrengthEnv({"strength_threshold": 50.0})
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(
        np.array([0.4, 0.3], dtype=np.float32)
    )

    assert info["predicted_strength"] == pytest.approx(70.0)
    assert reward == pytest.approx(1.0)  # (70-50)/10 = 2.0 clipped to reward_max=1.0
    assert terminated is True and truncated is False
```

- [x] **Step 2: Run the new tests**

Run: `uv run pytest tests/test_toy_problem_extension.py -v`
Expected: 3 PASS. If `build_arg_parser` rejects `toy_strength`, the registry wiring regressed — STOP and report BLOCKED (do not modify shared files to make it pass; that would defeat the acceptance proof).

- [x] **Step 3: Prove zero shared-file edits**

Run: `git status --short`
Expected: only `?? tests/test_toy_problem_extension.py` (plus the standing `?? .claude/`, `?? CLAUDE.md`). Paste the output in your report.

- [x] **Step 4: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `133 passed`; ruff clean; mypy exit 0.

```bash
git add tests/test_toy_problem_extension.py
git commit -m "test: prove third-problem registration needs zero shared-file edits" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: User-facing docs — how-to guide + CONTEXT.md terms

**Files:**
- Create: `docs/how-to-add-a-problem.md`
- Modify: `CONTEXT.md` (append three terms to the Language section, before Example Dialogue)

- [x] **Step 1: Create `docs/how-to-add-a-problem.md`**

```markdown
# How to add a new optimization problem

The PPO harness is problem-agnostic: `--env <name>` resolves through
`problems/registry.py`, and everything problem-specific travels in a
`ProblemSpec`. Adding a problem requires **no edits to shared harness code**
(`train_ppo.py`, `training/ppo_common.py`, callbacks/eval plumbing). The
executable specification of this contract is
`tests/test_toy_problem_extension.py` — read it alongside this guide.

## The five pieces

1. **Prediction backend** — how a candidate becomes a predicted value.
   Implement the `PredictionBackend` protocol (`env/backends.py`):
   `predict(features: Mapping[str, float]) -> PredictionResult`, `close()`.
   - Committee model-package ZIPs: reuse `CommitteePackageBackend` as-is.
   - ANN / other surrogates: implement the protocol directly; do not inherit
     the committee ZIP format or the verilogae toolchain.
   - Uncertainties are diagnostic only — never let them into reward.
2. **Objective** — how a predicted value becomes reward/success. Reuse
   `ThresholdMaximizeObjective` or `NRMSEMinimizeObjective`
   (`env/objectives.py`) when the semantics match; otherwise add a class with
   the same shape (`RANKED_METRIC`, `RANKED_ORDER`, reward + success methods).
   Episode control (termination/truncation) belongs to your env, not the
   objective (ADR 0003).
3. **Environment** — a `gymnasium.Env` whose observations are finite-bound
   float32 Boxes in every reset mode. Accept your backend via env-config
   injection (see `MaterialHardnessEnv`'s `prediction_backend_cls` or
   `EEHEMTEnv_Measure_VDS`'s `simulator_factory` for the two established
   patterns) so tests can stub it.
4. **Training module** — a module exposing `add_env_args(parser, current_dir)`,
   `build_env_config(args)`, `build_ppo_config(args, *, num_learners,
   num_gpus_per_learner)` (delegate to
   `training.ppo_common.build_base_ppo_config`), `build_checkpoint_config()`,
   and a `<NAME>_WANDB_PROJECT` constant. `training/hardness_ppo.py` is the
   reference implementation (~90 lines).
5. **Registration** — build a `ProblemSpec` and call
   `problems.registry.register(spec)` (see `problems/hardness.py`). Source
   `checkpoint_metric`/`checkpoint_order` from your objective class so the
   metric name has a single home.

## Checklist

- [x] Backend implements the protocol; committee ZIPs go in `env/<problem>/`,
      input data in `data/<problem>/` (both git-ignored; add `PUT_*_HERE.txt`
      placeholders).
- [x] Env-var defaults for your hyperparameters follow the existing pattern
      (`os.getenv` fallbacks in `add_env_args`); document them in `.env`.
- [x] Tests stub the backend through your injection seam (no model artifact
      or GPU needed — see `tests/conftest.py` and the existing env tests).
- [x] `uv run pytest && uv run ruff check . && uv run mypy .` all green
      (always via `uv run`; the global mypy lacks venv packages).
- [x] New layer-boundary decisions recorded in `docs/adr/` and
      `.codebase-memory/adr.md`.

## Worked example

`tests/test_toy_problem_extension.py` registers a complete toy problem
(deterministic backend + `ThresholdMaximizeObjective` + 2-action env +
training module) from test code only, then proves the CLI accepts
`--env toy_strength` and the generic PPO assembly builds its config. Copy its
shape, swap in your real backend/env, move the module under `problems/`, and
register it from `problems/__init__.py`.
```

- [x] **Step 2: Append three terms to CONTEXT.md**

Insert before the `## Example Dialogue` heading, matching the existing term format exactly:

```markdown
**Prediction Backend**:
The component that turns one candidate (an EEHEMT modelcard or a Material Composition) into predicted values, behind the `PredictionBackend` protocol. Physics simulators, committee model packages, and ANN surrogates are all Prediction Backends.
_Avoid_: the model (ambiguous), inference script

**Objective Strategy**:
The component that turns a predicted value into reward and success, and names the checkpoint-ranking metric. The NRMSE Objective and the Hardness Objective are realized by `NRMSEMinimizeObjective` and `ThresholdMaximizeObjective`. Episode control stays in the environment.
_Avoid_: reward function (partial), fitness

**Problem Spec**:
The registration record that binds one optimization problem's environment, training assembly, W&B project, and checkpoint metric under a `--env` name. Adding a problem means registering a Problem Spec, not editing the harness.
_Avoid_: env entry, config block
```

- [x] **Step 3: Full trio, commit**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `133 passed`; ruff clean; mypy exit 0.

```bash
git add docs/how-to-add-a-problem.md CONTEXT.md
git commit -m "docs: add problem-extension guide and ubiquitous-language terms" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Plan wrap-up verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-23-p3-plumbing-toy-problem-docs.md` (tick checkboxes)

- [x] **Step 1: Full verification trio**

```bash
uv run pytest && uv run ruff check . && uv run mypy .
```

Expected: `133 passed`, ruff clean, mypy exit 0.

- [x] **Step 2: Confirm untracked hygiene**

`git status --short` shows only `?? .claude/` and `?? CLAUDE.md`.

- [x] **Step 3: Tick all checkboxes in this plan and commit**

```bash
git add docs/superpowers/plans/2026-07-23-p3-plumbing-toy-problem-docs.md
git commit -m "docs: mark P3 plumbing/toy-problem/docs plan complete" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
