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
   reference implementation (~110 lines).
5. **Registration** — build a `ProblemSpec` and call
   `problems.registry.register(spec)` (see `problems/hardness.py`). Source
   `checkpoint_metric`/`checkpoint_order` from your objective class so the
   metric name has a single home.

## Checklist

- [ ] Backend implements the protocol; committee ZIPs go in `env/<problem>/`,
      input data in `data/<problem>/` (both git-ignored; add `PUT_*_HERE.txt`
      placeholders).
- [ ] Env-var defaults for your hyperparameters follow the existing pattern
      (`os.getenv` fallbacks in `add_env_args`); document them in `.env`.
- [ ] Tests stub the backend through your injection seam (no model artifact
      or GPU needed — see `tests/conftest.py` and the existing env tests).
- [ ] `uv run pytest && uv run ruff check . && uv run mypy .` all green
      (always via `uv run`; the global mypy lacks venv packages).
- [ ] New layer-boundary decisions recorded in `docs/adr/` and
      `.codebase-memory/adr.md`.

## Worked example

`tests/test_toy_problem_extension.py` registers a complete toy problem
(deterministic backend + `ThresholdMaximizeObjective` + 2-action env +
training module) from test code only, then proves the CLI accepts
`--env toy_strength` and the generic PPO assembly builds its config. Copy its
shape, swap in your real backend/env, move the module under `problems/`, and
register it from `problems/__init__.py`.
