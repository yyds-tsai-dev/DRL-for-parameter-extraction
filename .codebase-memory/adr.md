# ADR Update: Parameter Extraction Automation Boundaries

## Decision
The active PPO flow is split into stable boundaries:

- `env/parameter_flow.py` owns parameter specs, measured curve loading, EEHEMT simulation, and arcsinh Huber metric/reward scaling.
- `env/eehemt_env.py` owns Gymnasium/RLlib episode state, action application, observation assembly, reward/termination wiring, and legal observation guards.
- `evaluation/metrics.py` owns NRMSE/RMSE/RMSPE calculations for reporting and for the active NRMSE objective.
- `utils/plot.py` owns training-time curve visualization and uses the environment Vds values as the curve-condition axis.

The active PPO environment uses monotonic reward scaling over the NRMSE objective:

`reward = clip(-log10((NRMSE / 100) + EPSILON), REWARD_MIN, REWARD_MAX)`

Termination is based on NRMSE:

`terminated = NRMSE < NRMSE_THRESHOLD`

Tune checkpoints are ranked by lowest `env_runners/min_nrmse`.

IR-drop solves use multi-start continuation. Each Vds curve tries the previous
accepted Vds solution, a zero-current start, and a fixed-point warmup fallback,
accepting `ier == 1` or a residual below `IR_DROP_RESIDUAL_TOL`.

Observations must be legal Gymnasium `float32` arrays for both default and `random_init + reduce_obs_err_dim` resets. Error-observation bounds are finite via `OBS_ERR_BOUND` so RLlib/Gym checks do not rely on infinite Box limits.

Training defaults to `OBSERVATION_FILTER=NoFilter`. The older RLlib `MeanStdFilter` may still be selected explicitly, but it can transform reset observations into values/dtypes outside the declared Gymnasium Box under the current new RLlib API stack. Do not make it the default without adding a contract-preserving observation encoder or wrapper.

Ray runtime packaging excludes local state such as `.git/`, `.venv/`, caches, results, and archived demo files so training workers do not upload stale or bulky project state.

## Reason
This keeps reward, termination, and policy quality in the same metric space while making the PPO reward signal easier to learn from than raw `-NRMSE`. The arcsinh Huber loss remains exposed in `info["arcsinh_huber_loss"]` for diagnostics, but it is no longer the success objective.

The finite observation guard avoids subtle RLlib startup warnings and makes the env contract testable. NRMSE is now intentionally part of the environment reward because the project objective is lowest linear-current NRMSE, not lowest arcsinh Huber fit.

## Current defaults
`REWARD_MIN=-5.0`, `REWARD_MAX=5.0`, `NRMSE_THRESHOLD=10.0`, `ARCSINH_HUBER_THRESHOLD=1e-5` for diagnostics/backward compatibility, `OBS_ERR_BOUND=1e6`, and `OBSERVATION_FILTER=NoFilter` are the current defaults.
`IR_DROP_RESIDUAL_TOL=1e-8` is the current residual tolerance for accepting
solver results that SciPy reports as not fully converged.

## Guardrail
Do not make arcsinh Huber loss the primary reward or termination criterion unless the project objective changes again. It may be logged or used in diagnostics, but the active policy objective is NRMSE.

Do not restore SAC/test/restore legacy entrypoints as-is. Restore/test should be rebuilt on top of the current boundaries.

# ADR Update: 0003 — Problem registry, prediction backends, and objective strategies

## Decision
The PPO harness now dispatches `--env <name>` through a `problems/` registry (`ProblemSpec` — frozen dataclass carrying training module, W&B project, checkpoint metric/order, and the four assembly callables) instead of hardcoded per-env branches; built-ins self-register on package import and `select_training_module` stays as a facade. `training/ppo_common.build_base_ppo_config` owns the shared PPOConfig chain, with problem modules delegating and injecting env class, env config, callback, and evaluation function. `env/backends.py` defines a `PredictionBackend` protocol (`predict(features) -> PredictionResult`, `close()`), with `CommitteePackageBackend` wrapping `InferenceModel`; `MaterialHardnessEnv` accepts `prediction_backend_cls` alongside the legacy `inference_model_cls` path, and `EEHEMTEnv_Measure_VDS` accepts `simulator_factory`, giving EEHEMT its first test seam. `env/objectives.py` adds `ThresholdMaximizeObjective` and `NRMSEMinimizeObjective`, which own reward math, success comparison, and the ranked-metric identity (`RANKED_METRIC`/`RANKED_ORDER`); checkpoint configs and problem specs now read the metric from the objective class instead of repeating the string literal.

**Refinement of the original design spec:** episode control (termination, truncation, solver-failure penalty, episode-best tracking) deliberately stays in the environments rather than moving into the objective classes. Forcing the episodic, solver-coupled EEHEMT problem and the single-step hardness search into one generic env/objective contract was judged an over-generalization risk.

## Reason
Adding a third problem previously required editing shared harness code in at least four places and copying an entire vertical slice, with checkpoint-metric names repeated as string literals across training configs, callbacks, and evaluation. Two independent architecture reviews (deep-reasoner, Codex) converged on the same remedy: a registry plus shared assembly plus backend/objective seams. A new problem now needs only a problem package registering a `ProblemSpec`, an env (or reuse), a backend adapter satisfying `PredictionBackend`, and an objective (reusing a built-in where semantics match) — no shared-file edits.

## Guardrail
Guardrails preserved and test-locked: EEHEMT NRMSE reward/termination semantics (ADR 0001), hardness six-fraction action space, `NoFilter` default, float32 finite-bound observations, and verbatim checkpoint-metric strings (`env_runners/max_predicted_hardness`, `env_runners/min_nrmse`), which now exist only in `env/objectives.py`. `success_rate_650` is still emitted for dashboard continuity even though the threshold is configurable (600 in the current `.env`); a threshold-agnostic `success_rate` metric is planned alongside evaluation/callback parameterization (Plan 3). See `docs/adr/0003-problem-registry-backends-objectives.md` for full detail.
