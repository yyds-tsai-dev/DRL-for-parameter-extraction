# Material Hardness RL Design

## Context

The repository currently trains PPO agents for EEHEMT parameter extraction against measured I-V Curve data. The new material hardness flow uses a packaged XGB inference model as a surrogate objective: given material composition fractions, predict hardness and train an RL policy to find a Feasible Material Composition whose predicted hardness is at least 650.

The hardness model package accepts eight fraction inputs: `frac_Al`, `frac_Cr`, `frac_Mn`, `frac_Fe`, `frac_Co`, `frac_Cu`, `frac_Ni`, and `frac_Mo`. The first hardness RL iteration controls only Al, Cr, Mn, Fe, Co, and Ni. Cu and Mo are fixed at zero.

## Goals

- Add a hardness RL path that reuses the existing PPO/Ray/Tune training flow.
- Keep `train_ppo.py` as the shared entrypoint after renaming `train_ppo_tune.py`.
- Default the shared entrypoint to `--env hardness`.
- Keep EEHEMT training available through `--env eehemt`.
- Use a single-step bandit-style hardness environment.
- Project raw policy actions into Feasible Material Compositions.
- Rank hardness checkpoints by the highest unclipped predicted hardness.
- Save hardness evaluation outputs as structured CSV/JSON rather than I-V curve plots.
- Use environment-specific W&B project names from the selected training module.
- Accept the W&B API key through CLI args instead of relying only on environment variables.
- Keep XGB hardness inference usable without TensorFlow installed.

## Non-Goals

- Do not train or modify the hardness inference model.
- Do not make uncertainty part of the first-pass reward.
- Do not allow Cu or Mo to enter the hardness RL action space.
- Do not convert EEHEMT training to the hardness state/action design.
- Do not remove the EEHEMT flow or its I-V curve evaluation output.
- Do not add multi-objective material constraints beyond the confirmed composition bounds.

## Architecture

The shared entrypoint will be renamed from `train_ppo_tune.py` to `train_ppo.py`. It will dispatch by an `--env` option whose default is `hardness`.

Planned module boundaries:

```text
train_ppo.py
training/ppo_common.py
training/hardness_ppo.py
training/eehemt_ppo.py
env/material_hardness_env.py
utils/composition_projection.py
evaluation/hardness_evaluation.py
```

`train_ppo.py` will own only CLI entry and dispatch. It will pre-parse `--env`, let the selected training module register domain-specific CLI arguments, then parse the full argument set. `training/ppo_common.py` will own shared PPO/Ray/Tune setup: common PPO hyperparameters, learner and GPU resolution, Ray runtime configuration, restore handling, W&B callback construction, and new-run `Tuner` construction. Environment-specific modules will provide env class, env config, callbacks, evaluation function, checkpoint config, experiment name, W&B project name, and domain-specific CLI arguments.

The existing `scripts/train_ppo.sh` file will keep its name, but it will call `train_ppo.py`. Because `--env` defaults to `hardness`, the script will run hardness training by default. EEHEMT training must pass `--env eehemt`.

## Hardness Environment

`MaterialHardnessEnv` will be a Gymnasium single-step environment.

The observation is fixed:

```text
observation_space = Box(low=0, high=0, shape=(1,))
observation = np.zeros(1, dtype=np.float32)
```

The action is a raw six-dimensional vector:

```text
action_space = Box(low=-1, high=1, shape=(6,))
```

Action order is:

```text
frac_Al, frac_Cr, frac_Mn, frac_Fe, frac_Co, frac_Ni
```

The action is not used directly as model input. It is projected into a Feasible Material Composition:

```text
0.05 <= each tunable fraction <= 0.35
sum(tunable fractions) = 1.0
frac_Cu = 0.0
frac_Mo = 0.0
```

The projection helper lives in `utils/composition_projection.py`. It should be described and implemented as bounded simplex projection, not as generic normalization. A simple divide-by-sum normalization is not acceptable because it can violate per-element bounds.

Each episode ends after one action:

```text
reset() -> fixed observation
step(action) -> projected composition -> inference -> reward/info -> terminated=True
```

## Reward

The first-pass reward is dense and based only on predicted hardness:

```text
reward = clip((predicted_hardness - 650) / 100, -3, 3)
```

Success is:

```text
predicted_hardness >= 650
```

Model uncertainty is emitted as diagnostic info and evaluation output only. It does not affect reward or success in this iteration.

The environment info should include at least:

```text
composition
predicted_hardness
uncertainty_hardness
reward_unclipped
is_success
```

## Inference Package

The cleaned inference package is integrated into repository modules rather than left as a standalone unpacked folder. The XGB model package remains under the hardness inference resources.

The env adapter must always supply all model features in the expected order, including fixed `frac_Cu` and `frac_Mo`. Missing feature errors should fail fast with a clear message because they indicate integration drift between the env and the model package.

## Training And Checkpointing

Hardness checkpoint ranking uses unclipped predicted hardness:

```text
checkpoint_score_attribute = "env_runners/max_predicted_hardness"
checkpoint_score_order = "max"
```

This avoids ranking by clipped reward, which saturates at high predicted hardness and can hide better candidates.

Hardness callbacks should track:

```text
predicted_hardness
max_predicted_hardness
uncertainty_hardness
best_composition
success_rate_650
```

EEHEMT callbacks and checkpoint ranking remain NRMSE-oriented.

W&B logging remains part of the shared training path, but project naming is environment-specific. The hardness module should use:

```text
PPO_for_material_hardness_optimization
```

The EEHEMT module should keep the existing EEHEMT-oriented project name:

```text
PPO_for_multi_I-V_curves_fitting_in_EEHEMT
```

`training/ppo_common.py` should build the `WandbLoggerCallback` from the selected module's project name and the shared CLI arg:

```text
--wandb_api_key
```

The arg should default to `WANDB_API_KEY` from the environment when omitted. This keeps existing environment-variable usage working while allowing explicit API-key injection from scripts or job launchers.

## Evaluation

Hardness evaluation will not call I-V curve plotting. It will save structured artifacts under a hardness-specific evaluation directory.

Each evaluation output should include:

```text
evaluation_index
training_iteration
best_composition
predicted_hardness
max_predicted_hardness
uncertainty_hardness
success_rate_650
```

CSV is useful for quick comparison across checkpoints. JSON is useful for preserving nested composition data without column-name ambiguity.

## Dependencies

Core dependencies keep the XGB inference path:

```text
joblib
scikit-learn
xgboost
openpyxl
```

TensorFlow moves out of core dependencies and into an optional extra:

```toml
[project.optional-dependencies]
keras-inference = ["tensorflow>=2.19.0"]
```

The existing inference utility modules already use lazy TensorFlow imports. The XGB hardness path should import and run without TensorFlow installed.

## Data Flow

1. `train_ppo.py` parses shared args and `--env`.
2. The selected training module registers env-specific args and builds env-specific wiring, including the W&B project name.
3. `training/ppo_common.py` builds shared PPO/Ray/Tune objects.
4. RLlib creates `MaterialHardnessEnv` for `--env hardness`.
5. The policy emits a raw six-dimensional action.
6. `utils/composition_projection.py` projects the action into a Feasible Material Composition.
7. `MaterialHardnessEnv` injects fixed Cu and Mo, calls `InferenceModel`, computes reward, and terminates the episode.
8. Hardness callbacks log max predicted hardness and best composition.
9. Hardness evaluation writes CSV/JSON artifacts.
10. Tune checkpoints rank by `env_runners/max_predicted_hardness`.

## Error Handling

- Invalid model package path raises `FileNotFoundError` during env construction.
- Missing model feature columns raise `ValueError` with the missing feature names.
- Projection must always return a finite composition satisfying bounds and sum constraints within a small numeric tolerance.
- Non-finite model predictions end the single-step episode with reward `-3` and diagnostic info.
- Evaluation file write failures should be logged with context and re-raised, matching the existing I-V evaluation policy of not silently losing official outputs.
- If TensorFlow-only model operations are requested without the optional extra, the error message should direct the user to install `keras-inference`.

## Testing

Focused tests should avoid full Ray Tune training.

Add or update tests for:

- Bounded simplex projection preserves lower bound, upper bound, and sum-to-one constraints.
- Projection handles out-of-range, negative, and already-feasible vectors.
- `MaterialHardnessEnv.reset()` returns the fixed observation inside the observation space.
- `MaterialHardnessEnv.step()` terminates after one step and returns reward/info consistent with a fake inference model.
- The env injects `frac_Cu = 0.0` and `frac_Mo = 0.0`.
- Dense reward clips to `[-3, 3]` but checkpoint metrics use unclipped predicted hardness.
- `train_ppo.py` defaults to `--env hardness` and dispatches `--env eehemt`.
- Hardness PPO config uses `env_runners/max_predicted_hardness` with `max` ordering.
- W&B callback construction uses the selected environment's project name.
- `--wandb_api_key` defaults from `WANDB_API_KEY` and can be overridden explicitly.
- `scripts/train_ppo.sh` calls `train_ppo.py`.
- Importing the XGB inference path does not require TensorFlow.

## Migration Notes

The rename from `train_ppo_tune.py` to `train_ppo.py` requires updating imports in tests and scripts. Any documentation or shell commands that mention `train_ppo_tune.py` should be updated. Existing EEHEMT users must pass `--env eehemt` if they rely on the shared entrypoint default.
