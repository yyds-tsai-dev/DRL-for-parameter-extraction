# Episode-Best NRMSE Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PPO training, evaluation plots, callbacks, and checkpoint ranking use Episode-Best NRMSE, then restart training with a 5% threshold, 500-step episodes, and 600 iterations.

**Architecture:** Keep the existing PPO environment and RLlib integration. Add episode-best snapshot state inside `EEHEMTEnv_Measure_VDS`, surface that state through episode `info`, teach evaluation and callbacks to prefer episode-best fields, then update configuration, docs, tests, and training script.

**Tech Stack:** Python 3.11, Gymnasium, NumPy, Ray RLlib PPO, pytest, shell scripts, Markdown ADR/spec docs.

---

## File Structure

- Modify `env/eehemt_env.py`: owns environment state, episode-best NRMSE tracking, and info payloads.
- Modify `utils/callbacks.py`: owns RLlib metric logging at episode boundaries.
- Modify `evaluation/iv_curve_evaluation.py`: owns driver-side evaluation plotting.
- Modify `tests/test_env_measure_vds.py`: verifies reset, improvement, non-improvement, and solver-failure episode-best behavior.
- Modify `tests/test_callbacks.py`: verifies callback logs `episode_best_nrmse` and global `min_nrmse`.
- Modify `tests/test_iv_curve_evaluation.py`: verifies evaluation plots use episode-best matrix first and final/env matrices as fallback.
- Modify `tests/test_train_ppo_config.py`: verifies checkpoint ranking remains `env_runners/min_nrmse`.
- Modify `.env`: sets `MAX_EPISODE_STEPS=500` and `NRMSE_THRESHOLD=5.0`.
- Modify `scripts/train_ppo.sh`: runs `--n_iterations 600`.
- Modify `docs/adr/0001-nrmse-objective-reward.md`: records the Episode-Best NRMSE decision and new training threshold/horizon.
- Leave IR-drop solver files unchanged.

---

### Task 1: Environment Episode-Best Snapshot

**Files:**
- Modify: `tests/test_env_measure_vds.py`
- Modify: `env/eehemt_env.py`

- [ ] **Step 1: Write failing reset and episode-end tests**

Append these tests to `tests/test_env_measure_vds.py`:

```python
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


def test_episode_end_info_includes_best_snapshot_even_when_final_is_worse(monkeypatch):
    monkeypatch.setenv("MAX_EPISODE_STEPS", "1")
    env = EEHEMTEnv_Measure_VDS(_env_config())
    _, reset_info = env.reset(seed=123)

    _, _, terminated, truncated, step_info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert terminated or truncated
    assert step_info["episode_best_nrmse"] <= max(reset_info["nrmse"], step_info["nrmse"])
    assert "episode_best_i_sim_current_matrix" in step_info
    assert "episode_best_key_params" in step_info
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
pytest tests/test_env_measure_vds.py::test_reset_initializes_episode_best_snapshot tests/test_env_measure_vds.py::test_episode_end_info_includes_best_snapshot_even_when_final_is_worse -q
```

Expected: both tests fail with missing `episode_best_*` keys.

- [ ] **Step 3: Add episode-best state helpers to the environment**

In `env/eehemt_env.py`, replace `_get_info(...)` with:

```python
    def _get_info(
        self,
        arcsinh_huber_loss: float,
        nrmse: float,
        ir_drop_solver_diagnostics: list[dict[str, object]] | None = None,
    ) -> dict:
        current_key_params = {
            name: self.current_params[name] for name in key_params_names
        }
        diagnostics = ir_drop_solver_diagnostics or []
        failures = [
            diagnostic
            for diagnostic in diagnostics
            if not bool(diagnostic.get("converged", True))
        ]
        info = {
            "arcsinh_huber_loss": arcsinh_huber_loss,
            "nrmse": nrmse,
            "current_key_params": current_key_params,
            "ir_drop_solver_converged": not failures,
            "ir_drop_solver_failures": failures,
        }
        if hasattr(self, "episode_best_nrmse"):
            info.update(self._get_episode_best_info())
        return info

    def _record_episode_best(
        self,
        *,
        arcsinh_huber_loss: float,
        nrmse: float,
        i_sim_current_matrix: np.ndarray,
    ) -> None:
        self.episode_best_arcsinh_huber_loss = float(arcsinh_huber_loss)
        self.episode_best_nrmse = float(nrmse)
        self.episode_best_i_sim_current_matrix = np.array(
            i_sim_current_matrix,
            copy=True,
        )
        self.episode_best_key_params = {
            name: float(self.current_params[name]) for name in key_params_names
        }

    def _maybe_record_episode_best(
        self,
        *,
        arcsinh_huber_loss: float,
        nrmse: float,
        i_sim_current_matrix: np.ndarray,
        solver_converged: bool,
    ) -> None:
        if not solver_converged:
            return
        if not hasattr(self, "episode_best_nrmse") or nrmse < self.episode_best_nrmse:
            self._record_episode_best(
                arcsinh_huber_loss=arcsinh_huber_loss,
                nrmse=nrmse,
                i_sim_current_matrix=i_sim_current_matrix,
            )

    def _get_episode_best_info(self) -> dict[str, object]:
        return {
            "episode_best_arcsinh_huber_loss": self.episode_best_arcsinh_huber_loss,
            "episode_best_nrmse": self.episode_best_nrmse,
            "episode_best_i_sim_current_matrix": self.episode_best_i_sim_current_matrix,
            "episode_best_key_params": self.episode_best_key_params,
        }
```

In `reset()`, after `init_nrmse = calculate_nrmse(...)`, insert:

```python
        self._record_episode_best(
            arcsinh_huber_loss=avg_init_loss,
            nrmse=init_nrmse,
            i_sim_current_matrix=init_i_sim_matrix,
        )
```

In `step()`, after `solver_converged = not solver_failures`, insert:

```python
        self._maybe_record_episode_best(
            arcsinh_huber_loss=current_loss,
            nrmse=current_nrmse,
            i_sim_current_matrix=all_i_sim_matrix,
            solver_converged=solver_converged,
        )
```

In `step()`, replace:

```python
        if terminated or truncated:
            info["i_sim_current_matrix"] = all_i_sim_matrix
```

with:

```python
        if terminated or truncated:
            info["i_sim_current_matrix"] = all_i_sim_matrix
            info.update(self._get_episode_best_info())
```

In `_get_plot_data_matrix()`, replace the existing matrix block with:

```python
        if hasattr(self, "episode_best_i_sim_current_matrix"):
            plot_data["episode_best_i_sim_current_matrix"] = (
                self.episode_best_i_sim_current_matrix
            )
        if hasattr(self, "last_i_sim_current_matrix"):
            plot_data["i_sim_current_matrix"] = self.last_i_sim_current_matrix
```

- [ ] **Step 4: Run environment tests**

Run:

```bash
pytest tests/test_env_measure_vds.py -q
```

Expected: all tests in `tests/test_env_measure_vds.py` pass.

- [ ] **Step 5: Commit environment snapshot work**

Run:

```bash
git add env/eehemt_env.py tests/test_env_measure_vds.py
git commit -m "feat: track episode-best NRMSE in environment"
```

---

### Task 2: Callback Metrics Use Episode-Best NRMSE

**Files:**
- Modify: `tests/test_callbacks.py`
- Modify: `utils/callbacks.py`

- [ ] **Step 1: Replace callback test with episode-best assertions**

Replace `test_training_metrics_callback_logs_final_nrmse()` in `tests/test_callbacks.py` with:

```python
def test_training_metrics_callback_logs_episode_best_nrmse():
    callback = TrainingMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        {
            "arcsinh_huber_loss": 1.23e-4,
            "episode_best_arcsinh_huber_loss": 9.87e-5,
            "nrmse": 8.9,
            "episode_best_nrmse": 4.56,
        }
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert ("episode_best_nrmse", 4.56, "mean") in metrics_logger.logged_values
    assert ("min_nrmse", 4.56, "mean") in metrics_logger.logged_values
    assert not any(key == "last_nrmse" for key, _, _ in metrics_logger.logged_values)
```

Append:

```python
def test_training_metrics_callback_falls_back_to_final_nrmse_for_old_infos():
    callback = TrainingMetricsCallback()
    metrics_logger = FakeMetricsLogger()
    episode = FakeEpisode(
        {
            "arcsinh_huber_loss": 1.23e-4,
            "nrmse": 7.89,
        }
    )

    callback.on_episode_end(
        episode=episode,
        env_runner=None,
        metrics_logger=metrics_logger,
        env=None,
        env_index=0,
        rl_module=None,
    )

    assert ("episode_best_nrmse", 7.89, "mean") in metrics_logger.logged_values
    assert ("min_nrmse", 7.89, "mean") in metrics_logger.logged_values
```

- [ ] **Step 2: Run callback tests and verify failure**

Run:

```bash
pytest tests/test_callbacks.py -q
```

Expected: tests fail because `utils/callbacks.py` still logs `last_nrmse`.

- [ ] **Step 3: Update callback logging**

In `utils/callbacks.py`, replace the arcsinh block inside `on_episode_end()` with:

```python
        fit_loss = last_info.get(
            "episode_best_arcsinh_huber_loss",
            last_info.get("arcsinh_huber_loss"),
        )
        if fit_loss is not None:
            if fit_loss < self.min_arcsinh_huber_loss:
                self.min_arcsinh_huber_loss = fit_loss

            metrics_logger.log_value(
                "episode_best_arcsinh_huber_loss",
                fit_loss,
                reduce="mean",
            )
            metrics_logger.log_value(
                "min_arcsinh_huber_loss",
                self.min_arcsinh_huber_loss,
                reduce="mean",
            )
            logger.info(
                "Episode-best arcsinh Huber loss: %.6g; "
                "Min arcsinh Huber loss: %.6g",
                fit_loss,
                self.min_arcsinh_huber_loss,
            )
```

Replace the NRMSE block inside `on_episode_end()` with:

```python
        nrmse = last_info.get("episode_best_nrmse", last_info.get("nrmse"))
        if nrmse is not None:
            if nrmse < self.min_nrmse:
                self.min_nrmse = nrmse

            metrics_logger.log_value(
                "episode_best_nrmse",
                nrmse,
                reduce="mean",
            )
            metrics_logger.log_value(
                "min_nrmse",
                self.min_nrmse,
                reduce="mean",
            )
            logger.info(
                "Episode-best NRMSE: %.6g; Min NRMSE: %.6g",
                nrmse,
                self.min_nrmse,
            )
```

- [ ] **Step 4: Run callback tests**

Run:

```bash
pytest tests/test_callbacks.py -q
```

Expected: all callback tests pass.

- [ ] **Step 5: Commit callback work**

Run:

```bash
git add utils/callbacks.py tests/test_callbacks.py
git commit -m "feat: log episode-best NRMSE metrics"
```

---

### Task 3: Evaluation Plot Prefers Episode-Best Matrix

**Files:**
- Modify: `tests/test_iv_curve_evaluation.py`
- Modify: `evaluation/iv_curve_evaluation.py`

- [ ] **Step 1: Add evaluation preference test**

Append this test to `tests/test_iv_curve_evaluation.py`:

```python
def test_evaluate_and_plot_iv_curve_prefers_episode_best_matrix(monkeypatch):
    final_matrix = np.array(
        [
            [1.1e-3, 2.1e-3],
            [2.1e-3, 3.1e-3],
        ]
    )
    best_matrix = np.array(
        [
            [0.9e-3, 1.9e-3],
            [1.9e-3, 2.9e-3],
        ]
    )
    episode = FakeEpisode(
        infos=[
            {
                "arcsinh_huber_loss": 0.00456,
                "episode_best_arcsinh_huber_loss": 0.00123,
                "i_sim_current_matrix": final_matrix,
                "episode_best_i_sim_current_matrix": best_matrix,
            }
        ],
        env_steps=11,
        agent_steps=22,
    )
    runner = FakeEnvRunner(
        episodes=[episode],
        metrics={"runner_metric": 2},
        env=FakeEnv(curve_condition_values=[0.1, 0.2]),
    )
    eval_workers = FakeEvalWorkerGroup(runner)
    algorithm = FakeAlgorithm()
    save_calls = []
    monkeypatch.setattr(
        "evaluation.iv_curve_evaluation.save_evaluation_iv_curves",
        lambda **kwargs: save_calls.append(kwargs),
    )

    evaluate_and_plot_iv_curve(algorithm, eval_workers)

    assert len(save_calls) == 1
    assert save_calls[0]["plot_data"]["i_sim_current_matrix"] is best_matrix
    assert save_calls[0]["fit_loss"] == 0.00123
```

- [ ] **Step 2: Run evaluation preference test and verify failure**

Run:

```bash
pytest tests/test_iv_curve_evaluation.py::test_evaluate_and_plot_iv_curve_prefers_episode_best_matrix -q
```

Expected: test fails because evaluation still uses `i_sim_current_matrix` first.

- [ ] **Step 3: Update evaluation matrix and fit-loss selection**

In `evaluation/iv_curve_evaluation.py`, replace:

```python
    i_sim_current_matrix = final_info.get("i_sim_current_matrix")
    if i_sim_current_matrix is None:
        i_sim_current_matrix = plot_data.get("i_sim_current_matrix")
```

with:

```python
    i_sim_current_matrix = final_info.get("episode_best_i_sim_current_matrix")
    if i_sim_current_matrix is None:
        i_sim_current_matrix = final_info.get("i_sim_current_matrix")
    if i_sim_current_matrix is None:
        i_sim_current_matrix = plot_data.get("episode_best_i_sim_current_matrix")
    if i_sim_current_matrix is None:
        i_sim_current_matrix = plot_data.get("i_sim_current_matrix")
```

Replace:

```python
    fit_loss = final_info.get("arcsinh_huber_loss")
```

with:

```python
    fit_loss = final_info.get(
        "episode_best_arcsinh_huber_loss",
        final_info.get("arcsinh_huber_loss"),
    )
```

- [ ] **Step 4: Run evaluation tests**

Run:

```bash
pytest tests/test_iv_curve_evaluation.py -q
```

Expected: all evaluation tests pass.

- [ ] **Step 5: Commit evaluation work**

Run:

```bash
git add evaluation/iv_curve_evaluation.py tests/test_iv_curve_evaluation.py
git commit -m "feat: plot episode-best evaluation curves"
```

---

### Task 4: Training Configuration and ADR

**Files:**
- Modify: `.env`
- Modify: `scripts/train_ppo.sh`
- Modify: `docs/adr/0001-nrmse-objective-reward.md`
- Modify: `tests/test_train_ppo_config.py`

- [ ] **Step 1: Add script iteration test**

Append this test to `tests/test_train_ppo_config.py`:

```python
def test_train_ppo_script_requests_600_iterations():
    with open("scripts/train_ppo.sh", encoding="utf-8") as script_file:
        script = script_file.read()

    assert "--n_iterations 600" in script
    assert "--random_init" in script
    assert "--reduce_obs_err_dim" in script
```

- [ ] **Step 2: Run script test and verify failure**

Run:

```bash
pytest tests/test_train_ppo_config.py::test_train_ppo_script_requests_600_iterations -q
```

Expected: test fails because the script still requests `300` iterations.

- [ ] **Step 3: Update `.env` and training script**

In `.env`, replace:

```dotenv
MAX_EPISODE_STEPS=100
```

with:

```dotenv
MAX_EPISODE_STEPS=500
```

Replace:

```dotenv
NRMSE_THRESHOLD=10.0
```

with:

```dotenv
NRMSE_THRESHOLD=5.0
```

In `scripts/train_ppo.sh`, replace:

```bash
python train_ppo_tune.py --n_iterations 300 --random_init --reduce_obs_err_dim
```

with:

```bash
python train_ppo_tune.py --n_iterations 600 --random_init --reduce_obs_err_dim
```

- [ ] **Step 4: Update ADR**

Replace the first paragraph of `docs/adr/0001-nrmse-objective-reward.md` with:

```markdown
The active PPO parameter extraction flow optimizes normalized root mean squared error in linear current space. Reward is `clip(-log10((NRMSE / 100) + EPSILON), REWARD_MIN, REWARD_MAX)`. Training interpretation, evaluation plots, and checkpoints are based on Episode-Best NRMSE: the lowest NRMSE reached within a policy episode. Success termination uses `NRMSE < NRMSE_THRESHOLD`, with the active training threshold lowered to `5.0`, and checkpoints are ranked by lowest `env_runners/min_nrmse`; arcsinh Huber loss remains a diagnostic metric rather than the primary objective.
```

Append this bullet under `## Consequences`:

```markdown
Evaluation curves now show the episode-best simulated I-V Curve rather than the final step when those differ. This makes saved plots match the checkpoint metric and avoids hiding short-lived low-NRMSE states.
```

- [ ] **Step 5: Run config tests**

Run:

```bash
pytest tests/test_train_ppo_config.py -q
```

Expected: all config tests pass.

- [ ] **Step 6: Commit configuration and ADR**

Run:

```bash
git add .env scripts/train_ppo.sh docs/adr/0001-nrmse-objective-reward.md tests/test_train_ppo_config.py
git commit -m "chore: configure episode-best NRMSE training run"
```

---

### Task 5: Full Verification and Training Restart

**Files:**
- Read: `result/ckpt/`
- Run: `scripts/train_ppo.sh`

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
pytest \
  tests/test_env_measure_vds.py \
  tests/test_callbacks.py \
  tests/test_iv_curve_evaluation.py \
  tests/test_train_ppo_config.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Check git status before training**

Run:

```bash
git status --short
```

Expected: only pre-existing unrelated worktree changes remain, such as `.gitignore`, `CONTEXT.md`, `Model-Inference.7z`, or `model-inference/`. No modified files from Tasks 1-4 should remain unstaged.

- [ ] **Step 3: Start training**

Run:

```bash
bash scripts/train_ppo.sh
```

Expected: training starts with `--n_iterations 600`, prints environment setup with `MAX_EPISODE_STEPS=500` and `NRMSE_THRESHOLD=5.0` behavior through configuration, and creates a new run under `result/ckpt/EEHEMT_PPO/`.

- [ ] **Step 4: Monitor first progress output**

Run this in a second terminal if training is still running:

```bash
find result/ckpt/EEHEMT_PPO -maxdepth 2 -name progress.csv -printf '%T@ %p\n' | sort -nr | head -1
```

Expected: the newest `progress.csv` belongs to the newly started run.

- [ ] **Step 5: Verify metrics columns after progress exists**

Run after the new `progress.csv` has at least one row:

```bash
python - <<'PY'
import csv
from pathlib import Path

progress_files = sorted(
    Path("result/ckpt/EEHEMT_PPO").glob("*/progress.csv"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
progress = progress_files[0]
with progress.open(newline="") as handle:
    reader = csv.DictReader(handle)
    first = next(reader)

print(progress)
print("episode_best_nrmse" in first)
print("env_runners/episode_best_nrmse" in first)
print("env_runners/min_nrmse" in first)
PY
```

Expected: the newest progress path prints, `env_runners/episode_best_nrmse` prints `True`, and `env_runners/min_nrmse` prints `True`.

---

## Self-Review

- Spec coverage: threshold, episode horizon, training iterations, episode-best environment tracking, evaluation plot preference, callback metric semantics, ADR update, and test execution are all covered.
- Placeholder scan: no placeholder tasks or unresolved requirements remain.
- Type consistency: episode-best keys are consistently named `episode_best_nrmse`, `episode_best_arcsinh_huber_loss`, `episode_best_i_sim_current_matrix`, and `episode_best_key_params`; global checkpoint metric remains `min_nrmse`.
