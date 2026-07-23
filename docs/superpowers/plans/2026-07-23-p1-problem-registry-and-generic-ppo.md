# P1: Problem Registry + Generic PPO Assembly — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded `--env` dispatch with a problem registry and deduplicate the PPO config chain, so problem selection is data-driven — with zero behavior change.

**Architecture:** New `problems/` package holds a frozen `ProblemSpec` dataclass and a module-level registry; `train_ppo.py` and `training/ppo_common.py` consult the registry instead of hardcoded if/elif chains and duplicated `choices` lists. `training/ppo_common.build_base_ppo_config` absorbs the PPO chain duplicated between `training/eehemt_ppo.py` and `training/hardness_ppo.py`. Existing public functions stay as thin facades so every existing test keeps passing. Tooling hygiene (mypy.ini, ruff config, lint errors) is fixed first so the verification trio is trustworthy.

**Tech Stack:** Python 3.11, uv, pytest, Ray RLlib (`PPOConfig`), argparse, dataclasses.

**Design spec:** `docs/superpowers/specs/2026-07-23-pluggable-problem-architecture-design.md`

## Global Constraints

- CLI stays backward compatible: `--env eehemt|hardness`, default `hardness`; all existing flags and env-var fallbacks unchanged.
- Checkpoint metric strings verbatim: `env_runners/min_nrmse` (order `min`), `env_runners/max_predicted_hardness` (order `max`).
- W&B project strings verbatim: `PPO_for_material_hardness_optimization`, `PPO_for_multi_I-V_curves_fitting_in_EEHEMT`.
- `ValueError` message for unknown env verbatim: `Unsupported training environment: {name}`.
- `OBSERVATION_FILTER` default stays `NoFilter`.
- Run tests with `uv run pytest`, lint with `uv run ruff check .`, types with `uv run mypy .` (after Task 1 adds both to the dev group). Full suite green after every task.
- Every commit message ends with the trailer line: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Work on branch `pluggable-problem-architecture`. Never commit `.claude/` or `CLAUDE.md` (untracked local files — leave them untracked).
- Do not modify: `env/material_hardness_env.py`, `env/eehemt_env.py`, `utils/*callbacks*.py`, `evaluation/` (those belong to later plans).

---

### Task 1: Tooling hygiene — mypy.ini, ruff config, lint errors, dev deps

**Files:**
- Modify: `mypy.ini` (full rewrite)
- Modify: `pyproject.toml:37-39` (`[tool.ruff]` table)
- Modify: `env/file_process.py:1` and `env/file_process.py:130`
- Create: `utils/__init__.py`
- Modify: `uv.lock` (via `uv add`)

**Interfaces:**
- Produces: working `uv run ruff check .` and `uv run mypy .` commands used by every later task's verification steps.

Background: `mypy.ini` uses inline comments after boolean values; mypy's ini parser rejects them (`Not a boolean: True`), so every such option has silently been OFF. `utils/` has no `__init__.py`, so mypy sees `utils/dim_reduce.py` under two module names. ruff's top-level `ignore` key is deprecated, and there are 7 outstanding lint errors (all pre-existing on main). ruff and mypy are not project deps, so `uv sync` removes them from the venv.

- [x] **Step 1: Add ruff and mypy to the dev dependency group**

```bash
uv add --dev ruff mypy
```

Expected: `pyproject.toml` `[dependency-groups] dev` gains `ruff` and `mypy` entries; `uv.lock` updated; both tools now run via `uv run`.

- [x] **Step 2: Rewrite mypy.ini with comments on their own lines**

Replace the entire content of `mypy.ini` with:

```ini
[mypy]
# 若函式返回 Any 型別則發出警告
warn_return_any = True
# 允許變數在同一範圍內被重新定義為不同型別
allow_redefinition = True
# 檢查沒有型別註記的函式內部程式碼
check_untyped_defs = True
# 忽略因找不到模組型別存根 (stubs) 而產生的匯入錯誤
ignore_missing_imports = True
# 啟用增量模式，利用快取加速後續檢查
incremental = True
# 強制嚴格的 None 檢查
strict_optional = True
# 當 Mypy 本身崩潰時，顯示詳細的錯誤追蹤訊息
show_traceback = True
# 若 `# type: ignore` 註解已不再需要，則發出警告
warn_unused_ignores = True
# 若設定檔中有無法辨識或未使用的選項，則發出警告
warn_unused_configs = True
# 警告不可能被執行到的程式碼
warn_unreachable = True
# 在整個專案中停用 arg-type 檢查
disable_error_code = arg-type
# 排除測試與 log 目錄
exclude = ^(tests/|logs/)
```

- [x] **Step 3: Create `utils/__init__.py`**

Create the file with exactly this content (empty module docstring keeps ruff quiet):

```python
"""Utility helpers shared across the training harness."""
```

- [x] **Step 4: Migrate ruff config and add per-file ignores**

In `pyproject.toml`, replace:

```toml
[tool.ruff]
lint.extend-select = ["PTH"]
ignore = ["PTH"]
```

with:

```toml
[tool.ruff]

[tool.ruff.lint]
extend-select = ["PTH"]
ignore = ["PTH"]

[tool.ruff.lint.per-file-ignores]
"*.ipynb" = ["E402"]
"scripts/run_model_inference.py" = ["E402"]
```

(The `sys.path.insert` before `from env import InferenceModel` in `scripts/run_model_inference.py` is intentional, as are the notebook reload imports — per-file ignores, not code changes.)

- [x] **Step 5: Fix the two real lint errors in `env/file_process.py`**

Remove line 1 (`import shutil` — F401 unused). Remove line 130 (`original_features = metadata.get("features", {}).get("original_features") or model_features` — F841 assigned but never used; it is a pure read with no side effects).

- [x] **Step 6: Run ruff and confirm clean**

Run: `uv run ruff check .`
Expected: `All checks passed!` and NO deprecation warning about top-level `ignore`.

- [x] **Step 7: Run mypy; apply the fallback rule if needed**

Run: `uv run mypy .`

Expected: no `Not a boolean` config warnings, no `Source file found twice` error, no `import-not-found` for numpy/gymnasium/ray (venv now used + `ignore_missing_imports`).

**Fallback rule (deterministic):** the newly-activated strictness flags (`warn_return_any`, `check_untyped_defs`, `warn_unused_ignores`, `warn_unreachable`) were never actually enforced before. If mypy reports errors, do NOT fix code. Instead replace the strictness flags so the effective behavior matches the previously-running configuration, keeping only:

```ini
[mypy]
# 忽略因找不到模組型別存根 (stubs) 而產生的匯入錯誤
ignore_missing_imports = True
# 允許變數在同一範圍內被重新定義為不同型別
allow_redefinition = True
# 啟用增量模式
incremental = True
# 當 Mypy 本身崩潰時，顯示詳細的錯誤追蹤訊息
show_traceback = True
# 在整個專案中停用 arg-type 檢查
disable_error_code = arg-type
# 排除測試與 log 目錄
exclude = ^(tests/|logs/)
```

Re-run `uv run mypy .`; it must exit 0. In your task report, list which flags were dropped and paste the error inventory that forced the fallback (it becomes future cleanup input).

- [x] **Step 8: Run the full test suite**

Run: `uv run pytest`
Expected: `92 passed`

- [x] **Step 9: Commit**

```bash
git add mypy.ini pyproject.toml uv.lock env/file_process.py utils/__init__.py
git commit -m "chore: repair mypy/ruff configs and adopt tools as dev deps" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Characterization tests for dispatch contract gaps

**Files:**
- Modify: `tests/test_train_ppo_config.py` (append two tests)
- Modify: `tests/test_ppo_common.py` (append one test)

**Interfaces:**
- Consumes: `train_ppo.build_arg_parser`, `training.ppo_common.build_common_arg_parser` (existing).
- Produces: locked contract that Tasks 3–6 must preserve.

Existing tests already lock: default env `hardness`, `select_training_module` returning modules with the W&B constants, checkpoint metric strings, and full PPO wiring. Two gaps remain: rejection of unknown env names at the CLI, and the `NoFilter` default when the env var is unset.

- [x] **Step 1: Append to `tests/test_train_ppo_config.py`**

```python
def test_build_arg_parser_rejects_unknown_env():
    with pytest.raises(SystemExit):
        build_arg_parser("/project", ["--env", "nosuch"])
```

(`build_arg_parser`'s pre-parser declares `choices`; argparse exits with an error for an unknown value. `pytest` is already imported in this file.)

- [x] **Step 2: Append to `tests/test_ppo_common.py`**

```python
def test_common_parser_observation_filter_defaults_to_nofilter(monkeypatch):
    monkeypatch.delenv("OBSERVATION_FILTER", raising=False)
    parser = build_common_arg_parser("/project")

    args = parser.parse_args([])

    assert args.observation_filter == "NoFilter"
```

- [x] **Step 3: Run the new tests**

Run: `uv run pytest tests/test_train_ppo_config.py::test_build_arg_parser_rejects_unknown_env tests/test_ppo_common.py::test_common_parser_observation_filter_defaults_to_nofilter -v`
Expected: both PASS (they characterize current behavior; if either fails, STOP and report — the assumption inventory is wrong).

- [x] **Step 4: Run the full suite**

Run: `uv run pytest`
Expected: `94 passed`

- [x] **Step 5: Commit**

```bash
git add tests/test_train_ppo_config.py tests/test_ppo_common.py
git commit -m "test: characterize env dispatch rejection and NoFilter default" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `problems/registry.py` — ProblemSpec and registry (TDD)

**Files:**
- Create: `problems/registry.py`
- Create: `tests/test_problem_registry.py`

**Interfaces:**
- Produces (used by Tasks 4–5):
  - `ProblemSpec(name: str, module: ModuleType, wandb_project: str, checkpoint_metric: str, checkpoint_order: Literal["min", "max"], add_env_args: Callable, build_env_config: Callable, build_ppo_config: Callable, build_checkpoint_config: Callable)` — frozen dataclass.
  - `register(spec: ProblemSpec) -> None` — raises `ValueError(f"Problem already registered: {spec.name}")` on duplicates.
  - `get(name: str) -> ProblemSpec` — raises `ValueError(f"Unsupported training environment: {name}")` (message verbatim — it replaces the one in `train_ppo.py:27`).
  - `names() -> list[str]` — sorted registered names.
  - `clear() -> None` — test-only helper to reset state.

- [x] **Step 1: Write the failing tests**

Create `tests/test_problem_registry.py`:

```python
import types

import pytest

from problems import registry


def _dummy_spec(name="dummy"):
    module = types.ModuleType(f"{name}_module")
    return registry.ProblemSpec(
        name=name,
        module=module,
        wandb_project=f"{name}-project",
        checkpoint_metric=f"env_runners/{name}_metric",
        checkpoint_order="max",
        add_env_args=lambda parser, current_dir: parser,
        build_env_config=lambda args: {},
        build_ppo_config=lambda args, **kwargs: None,
        build_checkpoint_config=lambda: None,
    )


@pytest.fixture(autouse=True)
def _isolated_registry():
    saved = registry.snapshot()
    registry.clear()
    yield
    registry.restore(saved)


def test_register_and_get_roundtrip():
    spec = _dummy_spec()
    registry.register(spec)

    assert registry.get("dummy") is spec


def test_register_rejects_duplicate_name():
    registry.register(_dummy_spec())

    with pytest.raises(ValueError, match="Problem already registered: dummy"):
        registry.register(_dummy_spec())


def test_get_unknown_name_uses_legacy_error_message():
    with pytest.raises(
        ValueError, match="Unsupported training environment: nosuch"
    ):
        registry.get("nosuch")


def test_names_are_sorted():
    registry.register(_dummy_spec("zeta"))
    registry.register(_dummy_spec("alpha"))

    assert registry.names() == ["alpha", "zeta"]


def test_spec_is_immutable():
    spec = _dummy_spec()

    with pytest.raises(Exception):
        spec.name = "other"
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_problem_registry.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'problems'`

- [x] **Step 3: Write `problems/registry.py`**

```python
"""Problem registry: maps ``--env`` names to their training assembly parts.

A problem registers a :class:`ProblemSpec` once at import time (see
``problems/__init__.py``). ``train_ppo.py`` and ``training/ppo_common.py``
resolve everything problem-specific through this registry, so adding a new
problem requires no edits to shared harness code.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    module: ModuleType
    wandb_project: str
    checkpoint_metric: str
    checkpoint_order: Literal["min", "max"]
    add_env_args: Callable
    build_env_config: Callable
    build_ppo_config: Callable
    build_checkpoint_config: Callable


_REGISTRY: dict[str, ProblemSpec] = {}


def register(spec: ProblemSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Problem already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def get(name: str) -> ProblemSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unsupported training environment: {name}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)


def snapshot() -> dict[str, ProblemSpec]:
    """Test helper: capture current registrations."""
    return dict(_REGISTRY)


def clear() -> None:
    """Test helper: drop all registrations."""
    _REGISTRY.clear()


def restore(saved: dict[str, ProblemSpec]) -> None:
    """Test helper: reinstate a snapshot taken with :func:`snapshot`."""
    _REGISTRY.clear()
    _REGISTRY.update(saved)
```

Also create an EMPTY `problems/__init__.py` for now (Task 4 fills it):

```python
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_problem_registry.py -v`
Expected: 5 PASS

- [x] **Step 5: Full suite, lint, commit**

Run: `uv run pytest && uv run ruff check .`
Expected: `99 passed`; ruff clean.

```bash
git add problems/ tests/test_problem_registry.py
git commit -m "feat: add problem registry with immutable ProblemSpec" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Register the two built-in problems

**Files:**
- Create: `problems/hardness.py`
- Create: `problems/eehemt.py`
- Modify: `problems/__init__.py`
- Create: `tests/test_problem_builtin_specs.py`

**Interfaces:**
- Consumes: `problems.registry` (Task 3); `training.hardness_ppo`, `training.eehemt_ppo` (existing modules).
- Produces: `import problems` self-registers specs named `"hardness"` and `"eehemt"` whose callables ARE the existing module functions (identity, not copies).

- [x] **Step 1: Write the failing test**

Create `tests/test_problem_builtin_specs.py`:

```python
import problems  # noqa: F401  (import triggers self-registration)
from problems import registry
from training import eehemt_ppo, hardness_ppo


def test_builtin_problem_names():
    assert registry.names() == ["eehemt", "hardness"]


def test_hardness_spec_points_at_existing_module_parts():
    spec = registry.get("hardness")

    assert spec.module is hardness_ppo
    assert spec.wandb_project == "PPO_for_material_hardness_optimization"
    assert spec.checkpoint_metric == "env_runners/max_predicted_hardness"
    assert spec.checkpoint_order == "max"
    assert spec.add_env_args is hardness_ppo.add_env_args
    assert spec.build_env_config is hardness_ppo.build_env_config
    assert spec.build_ppo_config is hardness_ppo.build_ppo_config
    assert spec.build_checkpoint_config is hardness_ppo.build_checkpoint_config


def test_eehemt_spec_points_at_existing_module_parts():
    spec = registry.get("eehemt")

    assert spec.module is eehemt_ppo
    assert spec.wandb_project == "PPO_for_multi_I-V_curves_fitting_in_EEHEMT"
    assert spec.checkpoint_metric == "env_runners/min_nrmse"
    assert spec.checkpoint_order == "min"
    assert spec.add_env_args is eehemt_ppo.add_env_args
    assert spec.build_env_config is eehemt_ppo.build_env_config
    assert spec.build_ppo_config is eehemt_ppo.build_ppo_config
    assert spec.build_checkpoint_config is eehemt_ppo.build_checkpoint_config
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_problem_builtin_specs.py -v`
Expected: FAIL — `registry.names() == []` (nothing registered yet).

- [x] **Step 3: Write the two spec modules and the package init**

`problems/hardness.py`:

```python
"""Registration glue for the material-hardness optimization problem."""

from problems.registry import ProblemSpec
from training import hardness_ppo


def build_spec() -> ProblemSpec:
    return ProblemSpec(
        name="hardness",
        module=hardness_ppo,
        wandb_project=hardness_ppo.HARDNESS_WANDB_PROJECT,
        checkpoint_metric="env_runners/max_predicted_hardness",
        checkpoint_order="max",
        add_env_args=hardness_ppo.add_env_args,
        build_env_config=hardness_ppo.build_env_config,
        build_ppo_config=hardness_ppo.build_ppo_config,
        build_checkpoint_config=hardness_ppo.build_checkpoint_config,
    )
```

`problems/eehemt.py`:

```python
"""Registration glue for the EEHEMT parameter-extraction problem."""

from problems.registry import ProblemSpec
from training import eehemt_ppo


def build_spec() -> ProblemSpec:
    return ProblemSpec(
        name="eehemt",
        module=eehemt_ppo,
        wandb_project=eehemt_ppo.EEHEMT_WANDB_PROJECT,
        checkpoint_metric="env_runners/min_nrmse",
        checkpoint_order="min",
        add_env_args=eehemt_ppo.add_env_args,
        build_env_config=eehemt_ppo.build_env_config,
        build_ppo_config=eehemt_ppo.build_ppo_config,
        build_checkpoint_config=eehemt_ppo.build_checkpoint_config,
    )
```

`problems/__init__.py` (replace the empty file):

```python
"""Built-in problem registrations.

Importing this package registers every built-in problem exactly once.
Third-party problems call :func:`problems.registry.register` themselves.
"""

from problems import eehemt, hardness, registry

for _builder in (hardness.build_spec, eehemt.build_spec):
    _spec = _builder()
    if _spec.name not in registry.names():
        registry.register(_spec)

del _builder, _spec
```

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_problem_builtin_specs.py -v`
Expected: 3 PASS

- [x] **Step 5: Full suite, lint, commit**

Run: `uv run pytest && uv run ruff check .`
Expected: `102 passed`; ruff clean. (The registry-isolation fixture in `tests/test_problem_registry.py` restores builtin registrations via snapshot/restore, so ordering between test files cannot leak.)

```bash
git add problems/ tests/test_problem_builtin_specs.py
git commit -m "feat: self-register builtin hardness and eehemt problems" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Route `train_ppo.py` and `ppo_common.py` through the registry

**Files:**
- Modify: `train_ppo.py:22-27` (`select_training_module`), `train_ppo.py:35` (pre-parser choices), `train_ppo.py:64-69` (`_wandb_project_name`)
- Modify: `training/ppo_common.py:10` (`--env` choices)
- Test: existing `tests/test_train_ppo_config.py`, `tests/test_ppo_common.py` (no new tests — this task must keep every existing test green; that IS the acceptance)

**Interfaces:**
- Consumes: `problems.registry.get/names` (Tasks 3–4).
- Produces: `select_training_module(env_name) -> ModuleType` (unchanged signature — now a registry facade); `_wandb_project_name(env_name, training_module) -> str` (unchanged signature, module arg now unused but kept for call-site stability).

- [x] **Step 1: Edit `train_ppo.py`**

Add to the imports block (after `from ray.rllib.algorithms.ppo import PPO`):

```python
from problems import registry as problem_registry
```

Remove the now-unneeded direct import line `from training import eehemt_ppo, hardness_ppo` (the registry pulls those modules in).

Replace `select_training_module` (lines 22-27) with:

```python
def select_training_module(env_name: str) -> ModuleType:
    return problem_registry.get(env_name).module
```

(`registry.get` raises the verbatim legacy `ValueError` message for unknown names.)

Replace the pre-parser line (line 35) with:

```python
    pre_parser.add_argument(
        "--env", choices=problem_registry.names(), default="hardness"
    )
```

Replace `_wandb_project_name` (lines 64-69) with:

```python
def _wandb_project_name(env_name: str, training_module: ModuleType) -> str:
    del training_module  # kept for call-site stability; registry owns the mapping
    return problem_registry.get(env_name).wandb_project
```

- [x] **Step 2: Edit `training/ppo_common.py`**

Add import at the top (after `from ray.air.integrations.wandb import WandbLoggerCallback`):

```python
from problems import registry as problem_registry
```

Replace line 10 with:

```python
    parser.add_argument(
        "--env", choices=problem_registry.names(), default="hardness"
    )
```

(Import-cycle check, verified: `problems` imports `training.hardness_ppo`/`training.eehemt_ppo`, which import env/evaluation/utils modules — none of which import `training.ppo_common`. `training/__init__.py` is a bare package marker.)

- [x] **Step 3: Run the dispatch-related tests**

Run: `uv run pytest tests/test_train_ppo_config.py tests/test_ppo_common.py tests/test_problem_registry.py tests/test_problem_builtin_specs.py -v`
Expected: ALL PASS — especially `test_select_training_module_dispatches_by_env`, `test_train_ppo_defaults_to_hardness_env`, `test_build_arg_parser_rejects_unknown_env`, `test_common_parser_env_defaults_to_hardness_and_can_select_eehemt`.

- [x] **Step 4: Full suite, lint, mypy**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `102 passed`; ruff clean; mypy exit 0.

- [x] **Step 5: Commit**

```bash
git add train_ppo.py training/ppo_common.py
git commit -m "refactor: resolve env dispatch through the problem registry" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Generic PPO assembly — `build_base_ppo_config`

**Files:**
- Modify: `training/ppo_common.py` (add `build_base_ppo_config`)
- Modify: `training/hardness_ppo.py:62-103` (`build_ppo_config` delegates)
- Modify: `training/eehemt_ppo.py:81-122` (`build_ppo_config` delegates)
- Test: existing `tests/test_train_ppo_config.py::test_eehemt_ppo_config_wires_callbacks_and_driver_evaluation` and `::test_hardness_ppo_config_wires_callbacks_and_hardness_evaluation` (they assert every field of both configs — they are the identity proof)

**Interfaces:**
- Consumes: `argparse.Namespace` with the common-arg attributes (`num_env_runners`, `observation_filter`, `train_batch_size_per_learner`, `num_epochs`, `minibatch_size`, `lr`, `entropy_coeff`, `grad_clip`, `vf_loss_coeff`, `evaluation_interval`, `evaluation_num_env_runners`).
- Produces:

```python
def build_base_ppo_config(
    args,
    *,
    num_learners: int,
    num_gpus_per_learner: float,
    env_cls,
    env_config: dict,
    callbacks_class,
    custom_evaluation_function,
) -> PPOConfig
```

- [x] **Step 1: Add `build_base_ppo_config` to `training/ppo_common.py`**

Add imports at the top:

```python
from ray.rllib.algorithms.ppo import PPOConfig
```

Append the function:

```python
def build_base_ppo_config(
    args,
    *,
    num_learners: int,
    num_gpus_per_learner: float,
    env_cls,
    env_config: dict,
    callbacks_class,
    custom_evaluation_function,
) -> PPOConfig:
    """One PPO chain shared by every problem; specs inject the varying parts."""
    return (
        PPOConfig()
        .environment(
            env=env_cls,
            env_config=env_config,
        )
        .env_runners(
            num_env_runners=args.num_env_runners,
            observation_filter=args.observation_filter,
        )
        .training(
            train_batch_size_per_learner=args.train_batch_size_per_learner,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
            lr=args.lr * num_learners,
            entropy_coeff=args.entropy_coeff,  # type: ignore[arg-type]
            grad_clip=args.grad_clip,
            vf_loss_coeff=args.vf_loss_coeff,
            vf_clip_param=20.0,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=num_gpus_per_learner,
        )
        .callbacks(
            callbacks_class=callbacks_class,
        )
        .evaluation(
            evaluation_interval=args.evaluation_interval,
            evaluation_num_env_runners=args.evaluation_num_env_runners,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            custom_evaluation_function=custom_evaluation_function,
            evaluation_config={"explore": False},
        )
    )
```

- [x] **Step 2: Make `training/hardness_ppo.py::build_ppo_config` delegate**

IMPORTANT: the `build_base_ppo_config` import must be FUNCTION-LOCAL, not module-top. Reason (verified): after Task 5, `training/ppo_common.py` imports `problems` at module top, and `problems` imports `training.hardness_ppo` — a top-level import of `ppo_common` from `hardness_ppo` would close an import cycle during partial initialization.

Replace the entire `build_ppo_config` function (keep the signature; the `PPOConfig` import stays for the return annotation; do NOT add any new module-top import) with exactly:

```python
def build_ppo_config(
    args: argparse.Namespace,
    *,
    num_learners: int,
    num_gpus_per_learner: float,
) -> PPOConfig:
    from training.ppo_common import build_base_ppo_config

    return build_base_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        env_cls=MaterialHardnessEnv,
        env_config=build_env_config(args),
        callbacks_class=HardnessMetricsCallback,
        custom_evaluation_function=evaluate_and_save_hardness,
    )
```

- [x] **Step 3: Make `training/eehemt_ppo.py::build_ppo_config` delegate**

Replace the function body the same way:

```python
def build_ppo_config(
    args: argparse.Namespace,
    *,
    num_learners: int,
    num_gpus_per_learner: float,
) -> PPOConfig:
    from training.ppo_common import build_base_ppo_config

    return build_base_ppo_config(
        args,
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        env_cls=EEHEMTEnv_Measure_VDS,
        env_config=build_env_config(args),
        callbacks_class=TrainingMetricsCallback,
        custom_evaluation_function=evaluate_and_plot_iv_curve,
    )
```

- [x] **Step 4: Run the two wiring tests (field-by-field identity proof)**

Run: `uv run pytest tests/test_train_ppo_config.py -v`
Expected: ALL PASS, including both `*_wires_callbacks_*` tests asserting every config field (env class, callback class, eval fn, all numeric fields, `vf_clip_param == 20.0`).

- [x] **Step 5: Full suite, lint, mypy**

Run: `uv run pytest && uv run ruff check . && uv run mypy .`
Expected: `102 passed`; ruff clean; mypy exit 0.

- [x] **Step 6: Commit**

```bash
git add training/ppo_common.py training/hardness_ppo.py training/eehemt_ppo.py
git commit -m "refactor: extract shared PPO assembly into build_base_ppo_config" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Plan wrap-up verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-23-p1-problem-registry-and-generic-ppo.md` (tick checkboxes)

- [x] **Step 1: Full verification trio from a clean state**

```bash
uv run pytest && uv run ruff check . && uv run mypy .
```

Expected: `102 passed`, ruff clean, mypy exit 0.

- [x] **Step 2: Confirm no forbidden files were touched**

```bash
git log --stat main..HEAD -- env/material_hardness_env.py env/eehemt_env.py utils/callbacks.py utils/hardness_callbacks.py evaluation/
```

Expected: empty output (later plans own those files). `.claude/` and `CLAUDE.md` still untracked (`git status --short` shows them under `??`).

- [x] **Step 3: Tick all checkboxes in this plan and commit**

```bash
git add docs/superpowers/plans/2026-07-23-p1-problem-registry-and-generic-ppo.md
git commit -m "docs: mark P1 registry plan complete" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
