# ADR Update: Parameter Extraction Automation Boundaries

## Decision
The active PPO flow is split into stable boundaries:

- `env/parameter_flow.py` owns parameter specs, measured curve loading, EEHEMT simulation, and arcsinh Huber metric/reward scaling.
- `env/eehemt_env.py` owns Gymnasium/RLlib episode state, action application, observation assembly, reward/termination wiring, and legal observation guards.
- `evaluation/metrics.py` owns NRMSE/RMSE/RMSPE for offline reporting only.
- `utils/plot.py` owns training-time curve visualization and uses the environment Vds values as the curve-condition axis.

The active PPO environment uses monotonic reward scaling over the same arcsinh Huber loss objective:

`reward = clip(-log10(arcsinh_huber_loss + EPSILON), REWARD_MIN, REWARD_MAX)`

Termination remains based on raw loss:

`terminated = arcsinh_huber_loss < ARCSINH_HUBER_THRESHOLD`

Observations must be legal Gymnasium `float32` arrays for both default and `random_init + reduce_obs_err_dim` resets. Error-observation bounds are finite via `OBS_ERR_BOUND` so RLlib/Gym checks do not rely on infinite Box limits.

Training defaults to `OBSERVATION_FILTER=NoFilter`. The older RLlib `MeanStdFilter` may still be selected explicitly, but it can transform reset observations into values/dtypes outside the declared Gymnasium Box under the current new RLlib API stack. Do not make it the default without adding a contract-preserving observation encoder or wrapper.

Ray runtime packaging excludes local state such as `.git/`, `.venv/`, caches, results, and archived demo files so training workers do not upload stale or bulky project state.

## Reason
This keeps reward and termination in the same metric space while making the PPO reward signal larger and easier to learn from than raw `-loss`. The raw loss remains exposed in `info["arcsinh_huber_loss"]` for reporting and thresholding.

The finite observation guard avoids subtle RLlib startup warnings and makes the env contract testable. Keeping NRMSE/RMSE/RMSPE out of the environment prevents evaluation metrics from becoming hidden training objectives.

## Current defaults
`REWARD_MIN=-5.0`, `REWARD_MAX=5.0`, `ARCSINH_HUBER_THRESHOLD=1e-5`, `OBS_ERR_BOUND=1e6`, and `OBSERVATION_FILTER=NoFilter` are the current defaults.

## Guardrail
Do not reintroduce NRMSE/RMSE/RMSPE into env reward or termination. Those metrics live in `evaluation/metrics.py` for offline reporting only.

Do not restore SAC/test/restore legacy entrypoints as-is. Restore/test should be rebuilt on top of the current boundaries.