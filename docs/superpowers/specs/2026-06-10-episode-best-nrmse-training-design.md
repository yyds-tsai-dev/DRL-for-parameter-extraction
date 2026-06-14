# Episode-Best NRMSE Training Design

## Context

The current PPO parameter extraction flow optimizes the NRMSE Objective through a monotonic transformed reward. The latest 300-iteration run reached a much lower NRMSE during some episodes than it showed at the final episode step, so final-step metrics and final-step evaluation plots are no longer the right primary training signal.

The next training run should make Episode-Best NRMSE the reported policy-quality metric, extend the episode horizon for refinement, lower the success threshold, and run longer training.

## Goals

- Lower `NRMSE_THRESHOLD` from `10.0` to `5.0`.
- Raise `MAX_EPISODE_STEPS` from `100` to `500`.
- Raise `scripts/train_ppo.sh` training iterations from `300` to `600`.
- Use the episode-best simulated I-V Curve for evaluation plots.
- Record episode-best NRMSE in callbacks instead of final-step NRMSE.
- Keep checkpoint ranking on the lowest global `min_nrmse`, where `min_nrmse` now means the best Episode-Best NRMSE seen so far.
- Keep arcsinh Huber loss as diagnostic only.

## Non-Goals

- Do not change the NRMSE reward formula in this iteration.
- Do not change the IR-drop solver tolerance or acceptance policy.
- Do not add new tuning sweeps or optimizer changes.
- Do not change the measured dataset or selected key parameters.

## Design

The environment will track the best NRMSE reached within each episode. On reset, the initial simulated I-V Curve and its NRMSE seed the episode-best state. On every step, if the current NRMSE improves on that value and the IR-drop solver converged, the environment records the current NRMSE, arcsinh Huber loss, simulated current matrix, and key parameter values as the episode-best snapshot.

The existing success termination remains threshold-based, but the threshold is lowered to `5.0`. This keeps the "success" concept meaningful for very good fits while allowing refinement below the previous 10% stopping point. Episodes still truncate at `MAX_EPISODE_STEPS`, now `500`, or on solver failure.

When an episode ends, the info dictionary will include the episode-best snapshot. Evaluation plotting will prefer `episode_best_i_sim_current_matrix` and fall back to the final matrix only if the best snapshot is unavailable. This makes the saved Evaluation Curve represent the best fit achieved by that evaluation episode rather than whatever state happened to occur at the last step.

The training callback will log `episode_best_nrmse` and update the global `min_nrmse` from that value. It will no longer log `last_nrmse`. The existing checkpoint score attribute `env_runners/min_nrmse` remains valid, but its meaning changes to the lowest Episode-Best NRMSE observed during training.

## Data Flow

1. `reset()` simulates the initial I-V Curve and initializes the episode-best snapshot.
2. `step()` applies the action, simulates the new I-V Curve, computes NRMSE, and updates the episode-best snapshot on improvement.
3. At termination or truncation, `info` includes both current diagnostics and episode-best diagnostics.
4. The evaluation hook reads episode-best matrix data for plotting.
5. The callback logs episode-best NRMSE and global minimum NRMSE.
6. Tune checkpoints continue to rank by `env_runners/min_nrmse`.

## Error Handling

Solver failure remains a hard truncation with minimum reward. Solver-failed steps must not update the episode-best snapshot. If an older checkpoint or unexpected info payload lacks episode-best fields, plotting and callbacks fall back gracefully to current final-step fields so evaluation does not crash.

## Documentation

Update the NRMSE reward ADR to state that training interpretation, checkpoint ranking, and Evaluation Curves are based on Episode-Best NRMSE. The ADR should also record the lowered threshold and longer episode horizon as a refinement-oriented choice.

## Testing

- Add or update focused tests for environment episode-best tracking across reset, improvement, non-improvement, and episode end.
- Add or update callback tests to verify `episode_best_nrmse` and `min_nrmse` logging.
- Add or update evaluation tests to verify plotting prefers `episode_best_i_sim_current_matrix`.
- Run the relevant test subset before training.

## Training Run

After implementation and tests pass, restart PPO with `scripts/train_ppo.sh`. The run should use `--n_iterations 600 --random_init --reduce_obs_err_dim`, `NRMSE_THRESHOLD=5.0`, and `MAX_EPISODE_STEPS=500`.
