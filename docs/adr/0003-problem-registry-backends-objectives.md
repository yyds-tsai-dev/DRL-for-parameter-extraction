# 0003 — Problem registry, prediction backends, and objective strategies

Date: 2026-07-23
Status: accepted

## Context

The PPO harness supported two problems (`eehemt`, `hardness`) through
hardcoded `--env` dispatch, copy-pasted training assembly, reward/termination
logic inlined in each env's `step()`, and checkpoint-metric names repeated as
string literals across training configs, callbacks, and evaluation. Adding a
third problem (new material, objective, or prediction model) required editing
shared harness code in at least four places and copying an entire vertical
slice. Two independent architecture reviews (deep-reasoner, Codex) converged
on the same remedy.

## Decision

1. **Problem registry** (`problems/`): `--env <name>` resolves via
   `problems.registry` (`ProblemSpec` — frozen dataclass carrying the training
   module, W&B project, checkpoint metric/order, and the four assembly
   callables). Built-ins self-register on package import;
   `select_training_module` stays as a facade.
2. **Generic PPO assembly**: `training/ppo_common.build_base_ppo_config` owns
   the shared PPOConfig chain; problem modules delegate, injecting env class,
   env config, callback, and evaluation function.
3. **Prediction backends** (`env/backends.py`): a `PredictionBackend` protocol
   (`predict(features) -> PredictionResult`, `close()`) with
   `CommitteePackageBackend` wrapping `InferenceModel` via `predict_array` +
   declared `targets`. `MaterialHardnessEnv` accepts `prediction_backend_cls`
   alongside the legacy `inference_model_cls` path; both flow through one
   `step()` body via a value-reader indirection. `EEHEMTEnv_Measure_VDS`
   accepts `simulator_factory`, closing the injection asymmetry and giving
   EEHEMT its first test seam. Uncertainty remains diagnostic-only.
4. **Objective strategies** (`env/objectives.py`):
   `ThresholdMaximizeObjective` and `NRMSEMinimizeObjective` own reward math,
   success comparison, and the ranked-metric identity
   (`RANKED_METRIC`/`RANKED_ORDER`). Checkpoint configs and problem specs read
   the metric from the objective class — the string exists in one place.
   **Refinement of the original design spec:** episode control (termination,
   truncation, solver-failure penalty, episode-best tracking) deliberately
   stays in the envs. Forcing the episodic, solver-coupled EEHEMT problem and
   the single-step hardness search into one generic env/objective contract was
   judged an over-generalization risk.

## Consequences

- A new problem needs: a problem package registering a `ProblemSpec`, an env
  (or reuse), a backend adapter satisfying `PredictionBackend`, and an
  objective (reusing a built-in where semantics match). No shared-file edits.
- Guardrails preserved and test-locked: EEHEMT NRMSE reward/termination
  semantics (ADR 0001), hardness six-fraction action space, NoFilter default,
  float32 finite-bound observations, verbatim checkpoint-metric strings.
- `success_rate_650` is still emitted for dashboard continuity even though the
  threshold is configurable (600 in the current `.env`); a threshold-agnostic
  `success_rate` metric is planned alongside the evaluation/callback
  parameterization (Plan 3).
