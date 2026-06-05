# NRMSE Objective Reward

The active PPO parameter extraction flow now optimizes normalized root mean squared error in linear current space. Reward is `clip(-log10((NRMSE / 100) + EPSILON), REWARD_MIN, REWARD_MAX)`, success termination uses `NRMSE < NRMSE_THRESHOLD`, and checkpoints are ranked by lowest `env_runners/min_nrmse`; arcsinh Huber loss remains a diagnostic metric rather than the primary objective.

## Considered Options

- Keep arcsinh Huber loss as the reward: stable across current scales, but it can improve while linear-current NRMSE remains high.
- Use raw current-vector error directly: aligned with the data, but too high-dimensional and poorly scaled for PPO.
- Use transformed NRMSE as the reward: aligned with the actual objective while preserving a dense scalar reward signal.

## Consequences

Checkpoint selection, success thresholds, and training interpretation are based on NRMSE. Arcsinh Huber loss may still explain curve-shape or low-current behavior, but it must not be treated as the main fit objective.
