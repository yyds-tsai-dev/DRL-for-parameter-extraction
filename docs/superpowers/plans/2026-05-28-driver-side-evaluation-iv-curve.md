# Driver-Side Evaluation I-V Curve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save official I-V curve images from driver-side evaluation only, while splitting plotting out of the training metrics callback.

**Architecture:** `TrainingMetricsCallback` records training episode metrics only. `evaluation/iv_curve_evaluation.py` implements RLlib custom evaluation and calls a Ray-independent plotting save helper in `utils/plot.py`. Scoped logging is configured from `train_ppo_tune.py`.

**Tech Stack:** Python 3.11, Ray RLlib 2.48.0, matplotlib, pytest, Gymnasium.

---

## File Structure

- Create `utils/logging_config.py`: central logging setup for the active PPO path.
- Modify `utils/plot.py`: rename `PlotCurve` to `TrainingMetricsCallback`, remove callback PNG saving, add `save_evaluation_iv_curves`.
- Create `evaluation/iv_curve_evaluation.py`: driver-side custom evaluation function and small helper functions for extracting plot payloads.
- Modify `train_ppo_tune.py`: configure logging, use `TrainingMetricsCallback`, wire custom evaluation.
- Create or modify tests:
  - `tests/test_plot_evaluation_curves.py`
  - `tests/test_iv_curve_evaluation.py`
  - `tests/test_train_ppo_config.py`

## Task 1: Add Scoped Logging

**Files:**
- Create: `utils/logging_config.py`
- Modify: `train_ppo_tune.py`
- Test: no dedicated test; verified through import and existing tests

- [ ] **Step 1: Create logging helper**

Create `utils/logging_config.py` with:

```python
import logging
import os


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 2: Wire logging in training entrypoint**

In `train_ppo_tune.py`, import and call:

```python
from utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)
```

Call `configure_logging()` immediately after `load_dotenv()` inside `if __name__ == "__main__":`.

- [ ] **Step 3: Replace active entrypoint prints**

Replace `print(...)` status messages in `train_ppo_tune.py` with `logger.info(...)` for restore, new training, early stop, training complete, and final checkpoint messages.

- [ ] **Step 4: Run import smoke**

Run: `.venv/bin/python -m py_compile utils/logging_config.py train_ppo_tune.py`

Expected: exit code 0.

## Task 2: Split Training Metrics Callback From Plot Saving

**Files:**
- Modify: `utils/plot.py`
- Test: `tests/test_plot_evaluation_curves.py`

- [ ] **Step 1: Rename callback class**

Rename `class PlotCurve(DefaultCallbacks)` to `class TrainingMetricsCallback(DefaultCallbacks)`.

- [ ] **Step 2: Remove callback plotting state**

Remove `plot_dir`, `plot_cnt`, `curve_condition_values`, `vds`, and `plot_data` from the callback unless needed for metrics. `on_environment_created` should be removed if it only fetched static plot data for plotting.

- [ ] **Step 3: Keep metric behavior**

Keep `on_episode_start`, `on_episode_step`, and `on_episode_end` metric logging:

```python
metrics_logger.log_value("last_arcsinh_huber_loss", fit_loss, reduce="mean")
metrics_logger.log_value("min_arcsinh_huber_loss", self.min_arcsinh_huber_loss, reduce="mean")
metrics_logger.log_value(f"avg_{param_name}", avg_param_value, reduce="mean")
```

- [ ] **Step 4: Ensure callback never saves plots**

`TrainingMetricsCallback.on_episode_end` must not call `plot_all_condition_iv_curve` or any save helper.

- [ ] **Step 5: Add compatibility alias only if needed**

If any tests or imports still use `PlotCurve`, either update them or add:

```python
PlotCurve = TrainingMetricsCallback
```

Prefer updating active imports to `TrainingMetricsCallback`.

- [ ] **Step 6: Run compile smoke**

Run: `.venv/bin/python -m py_compile utils/plot.py`

Expected: exit code 0.

## Task 3: Add Evaluation Curve Save Helper

**Files:**
- Modify: `utils/plot.py`
- Create: `tests/test_plot_evaluation_curves.py`

- [ ] **Step 1: Write tests for output files**

Create `tests/test_plot_evaluation_curves.py` with a test that builds minimal `plot_data`, calls `save_evaluation_iv_curves(...)`, and asserts four files exist: historical linear, historical log, latest linear, latest log.

- [ ] **Step 2: Run test to verify failure**

Run: `.venv/bin/python -m pytest tests/test_plot_evaluation_curves.py -q`

Expected: fail because `save_evaluation_iv_curves` does not exist.

- [ ] **Step 3: Implement save helper**

Add to `utils/plot.py`:

```python
from pathlib import Path
import shutil


def _format_loss_for_filename(loss: float | None) -> str:
    if loss is None:
        return "unknown"
    return f"{loss:.3e}"


def save_evaluation_iv_curves(
    *,
    curve_condition_values: list,
    plot_data: dict,
    plot_dir: str,
    evaluation_index: int,
    training_iteration: int,
    fit_loss: float | None,
) -> dict[str, str]:
    output_dir = Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_label = _format_loss_for_filename(fit_loss)
    stem = (
        f"eval_{evaluation_index:06d}_"
        f"iter_{training_iteration:06d}_"
        f"loss_{loss_label}"
    )
    linear_path = output_dir / f"{stem}.png"
    log_path = output_dir / f"{stem}_log.png"
    plot_all_condition_iv_curve(
        curve_condition_values=curve_condition_values,
        plot_data=plot_data,
        plot_dir=str(output_dir),
        log_y=False,
        save_path=str(linear_path),
    )
    plot_all_condition_iv_curve(
        curve_condition_values=curve_condition_values,
        plot_data=plot_data,
        plot_dir=str(output_dir),
        log_y=True,
        save_path=str(log_path),
    )
    latest_linear = output_dir / "latest_eval.png"
    latest_log = output_dir / "latest_eval_log.png"
    shutil.copyfile(linear_path, latest_linear)
    shutil.copyfile(log_path, latest_log)
    return {
        "linear": str(linear_path),
        "log": str(log_path),
        "latest_linear": str(latest_linear),
        "latest_log": str(latest_log),
    }
```

- [ ] **Step 4: Update plot function signature**

Update `plot_all_condition_iv_curve(...)` to accept optional `save_path: str | None = None`. Preserve old generated filenames when `save_path` is `None`.

- [ ] **Step 5: Run helper tests**

Run: `.venv/bin/python -m pytest tests/test_plot_evaluation_curves.py -q`

Expected: pass.

## Task 4: Implement Driver-Side Custom Evaluation

**Files:**
- Create: `evaluation/iv_curve_evaluation.py`
- Create: `tests/test_iv_curve_evaluation.py`

- [ ] **Step 1: Write tests with fakes**

Create tests with fake algorithm, fake eval worker group, fake worker, fake env, and fake episode. Verify:

- returns `(eval_results, env_steps, agent_steps)`
- calls save helper once when final info has `i_sim_current_matrix`
- skips plotting and logs warning when final info lacks `i_sim_current_matrix`

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_iv_curve_evaluation.py -q`

Expected: fail because module does not exist.

- [ ] **Step 3: Implement extraction helpers**

Create helpers in `evaluation/iv_curve_evaluation.py`:

```python
def _episode_final_info(episode) -> dict:
    infos = getattr(episode, "infos", None) or []
    return infos[-1] if infos else {}


def _episode_env_steps(episode) -> int:
    value = getattr(episode, "env_steps", None)
    return int(value() if callable(value) else value or 0)


def _episode_agent_steps(episode) -> int:
    value = getattr(episode, "agent_steps", None)
    return int(value() if callable(value) else value or 0)
```

- [ ] **Step 4: Implement custom evaluation function**

Implement `evaluate_and_plot_iv_curve(algorithm, eval_workers)` using RLlib custom evaluation contract. It should sample one episode from evaluation workers, aggregate worker metrics with `algorithm.metrics.aggregate(...)`, read evaluation metrics via `algorithm.metrics.peek(...)`, extract the first episode payload, and call `save_evaluation_iv_curves(...)` when plot data is complete.

- [ ] **Step 5: Run evaluation tests**

Run: `.venv/bin/python -m pytest tests/test_iv_curve_evaluation.py -q`

Expected: pass.

## Task 5: Wire Training Config To Driver Evaluation

**Files:**
- Modify: `train_ppo_tune.py`
- Modify: `tests/test_train_ppo_config.py`

- [ ] **Step 1: Update imports**

Replace:

```python
from utils.plot import PlotCurve
```

with:

```python
from evaluation.iv_curve_evaluation import evaluate_and_plot_iv_curve
from utils.plot import TrainingMetricsCallback
```

- [ ] **Step 2: Update RLlib config**

Use:

```python
.callbacks(callbacks_class=TrainingMetricsCallback)
.evaluation(
    evaluation_interval=args.evaluation_interval,
    evaluation_num_env_runners=args.evaluation_num_env_runners,
    evaluation_duration=1,
    evaluation_duration_unit="episodes",
    custom_evaluation_function=evaluate_and_plot_iv_curve,
    evaluation_config={"explore": False},
)
```

- [ ] **Step 3: Update config tests**

Assert active config references `TrainingMetricsCallback` and `evaluate_and_plot_iv_curve`.

- [ ] **Step 4: Run config tests**

Run: `.venv/bin/python -m pytest tests/test_train_ppo_config.py -q`

Expected: pass.

## Task 6: Regression Suite And Cleanup

**Files:**
- Modify only files touched by earlier tasks if verification reveals issues.

- [ ] **Step 1: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_plot_evaluation_curves.py \
  tests/test_iv_curve_evaluation.py \
  tests/test_train_ppo_config.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run broader existing tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_parameter_flow.py tests/test_env_measure_vds.py tests/test_demo_helper.py -q
```

Expected: pass or report pre-existing failures with exact output.

- [ ] **Step 3: Inspect git diff**

Run: `git diff --stat` and `git diff --check`.

Expected: no whitespace errors; diff only touches planned files plus `CONTEXT.md` and the spec/plan docs.

