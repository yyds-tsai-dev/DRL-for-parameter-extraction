# Driver-Side Evaluation I-V Curve Design

## Goal

Official I-V curve images should be saved once per evaluation checkpoint from the Ray driver, not independently by parallel training environment runners.

## Decisions

- Official saved I-V curves are **evaluation curves**.
- Training environment runners must not write official I-V curve PNGs.
- `evaluation_interval` is the only frequency control for official I-V curve output.
- `PLOT_PERIOD` no longer controls official evaluation curve output.
- Historical evaluation curve files are never overwritten.
- `latest_eval.png` and `latest_eval_log.png` may be overwritten as convenience aliases.
- The active PPO path should use scoped Python logging instead of new `print(...)` output.

## Architecture

The current `PlotCurve` callback will be renamed and split. Its remaining responsibility is training episode metrics, so the new name will be `TrainingMetricsCallback`.

A new driver-side custom evaluation helper will live in `evaluation/iv_curve_evaluation.py`. It will run evaluation sampling, collect the final episode simulation matrix, collect static plot data, save the official I-V curve images, and return RLlib evaluation metrics using the custom evaluation function contract.

`utils/plot.py` will keep the pure plotting function and add a Ray-independent save helper for evaluation curve filenames and latest aliases.

## Data Flow

1. RLlib enters the custom evaluation function on the driver.
2. The helper samples one evaluation episode.
3. The final episode info provides `arcsinh_huber_loss`, `current_key_params`, and `i_sim_current_matrix`.
4. Static plot data comes from the evaluation environment via `_get_plot_data_matrix()` and `curve_condition_values` or `vds`.
5. The driver calls the plot save helper to write linear and log-scale images.
6. The custom evaluation function aggregates and returns RLlib metrics plus environment and agent step counts.

## Output Files

For evaluation index `1`, training iteration `2`, and loss `1.23e-04`, the helper writes:

- `eval_000001_iter_000002_loss_1.230e-04.png`
- `eval_000001_iter_000002_loss_1.230e-04_log.png`
- `latest_eval.png`
- `latest_eval_log.png`

The historical names are unique for each evaluation checkpoint. Only the latest aliases are overwritten.

## Error Handling

- Missing `i_sim_current_matrix` logs a warning and skips plotting; evaluation metrics still return.
- Missing static plot data logs a warning and skips plotting; evaluation metrics still return.
- `matplotlib.savefig` failures are logged with context and re-raised so training stops rather than silently losing official output.
- Training runner callback paths should avoid noisy per-episode skip logs.

## Logging Scope

Logging changes are limited to the active PPO path:

- `train_ppo_tune.py`
- `utils/plot.py`
- `evaluation/iv_curve_evaluation.py`
- limited `env/eehemt_env.py` changes only if needed

Demo scripts, notebooks, and legacy `mtds/` code are out of scope.

## Tests

Tests should avoid full Ray Tune training. Use focused unit tests for:

- evaluation curve filename generation and latest aliases
- custom evaluation behavior with fake algorithm, fake evaluation workers, and fake episodes
- callback split behavior so `TrainingMetricsCallback` no longer saves PNGs

