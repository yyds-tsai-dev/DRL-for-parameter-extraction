# DRL-on-parameter-extraction

[繁體中文](README.zh-TW.md)

A PPO training harness for optimization problems that wrap a prediction model.
It ships with two problems: extracting EEHEMT transistor model parameters by
fitting simulated I-V curves to measured data, and searching for alloy
compositions whose predicted hardness clears a threshold. The harness itself is
problem-agnostic. A problem registers itself, and the trainer looks it up by
name.

Built on Ray RLlib and Tune, with Weights & Biases logging.

## Setup

Requires Python 3.11 on Linux (the `verilogae` compiler that builds the EEHEMT
model only supports this combination). Dependencies live in `pyproject.toml`
and are managed with `uv`:

```bash
uv sync
```

Model artifacts and measurement data are not committed. Put the hardness model
ZIP in `src/env/hardness/` and input CSVs in `data/hardness/`; the `PUT_*_HERE.txt`
placeholders mark the spots. EEHEMT measurement data goes in `data/eehemt/`.

## Training

`train_ppo.py` is the single entry point. Pick a problem with `--env`:

```bash
uv run python train_ppo.py --env hardness
uv run python train_ppo.py --env eehemt
```

Configuration comes from `.env` (loaded at startup via python-dotenv), and CLI
flags override environment variables. Almost every hyperparameter has an
env-var default, so check `.env` before assuming a code default is what runs.
Pass `--restore_path` to resume a Tune run. `scripts/train_ppo.sh` wraps the
same command.

Each problem defines its own objective and checkpoint ranking:

- `eehemt` minimizes NRMSE in linear current space. Checkpoints rank by the
  lowest `env_runners/min_nrmse`.
- `hardness` maximizes predicted hardness against a threshold. Checkpoints
  rank by the highest `env_runners/max_predicted_hardness`. The committee
  model's uncertainty is logged as a diagnostic and never enters the reward.

## Adding a new problem

The two current problems are both plugins: any problem shaped like "search
for an input whose predicted value is good enough" fits. You write a few
small pieces, register them under a name, and
`train_ppo.py --env <name>` picks the new problem up. None of the shared
training code (`train_ppo.py`, `src/training/ppo_common.py`) needs to change.

The pieces are:

1. A prediction backend: code that takes one candidate solution and returns a
   predicted value. If your model is packaged as a ZIP like the hardness
   model, the existing `CommitteePackageBackend` already handles it. For
   anything else, write a small class with a `predict` and a `close` method;
   `src/env/backends.py` shows the shape.
2. An objective: the rule that turns a prediction into a reward and decides
   when the problem counts as solved. `src/env/objectives.py` has two ready
   to use: "push a value above a threshold" and "push an error below a
   threshold". Reuse one if it matches your goal. If neither fits, write a
   class with the same shape: the `RANKED_METRIC` and `RANKED_ORDER`
   constants plus the reward and success methods. When an episode ends is
   the environment's decision; keep it out of the objective.
3. An environment: a standard Gymnasium environment that wires the backend
   and objective together. `src/env/material_hardness_env.py` is an example
   to copy. Two requirements: observations must stay inside finite-bound
   float32 Boxes in every reset mode, and the backend must come in through
   the env config so tests can swap it for a stub. `MaterialHardnessEnv`'s
   `prediction_backend_cls` and `EEHEMTEnv_Measure_VDS`'s
   `simulator_factory` show the two existing ways.
4. A training module: declares the problem's command-line options and PPO
   settings. `src/training/hardness_ppo.py` is the reference, at about 90
   lines. The module provides `add_env_args(parser, current_dir)`,
   `build_env_config(args)`, `build_ppo_config(args, *, num_learners, num_gpus_per_learner)`, `build_checkpoint_config()`, and a
   `<NAME>_WANDB_PROJECT` constant.
5. Registration: a small file under `src/problems/` that gives the problem
   its `--env` name. Copy `problems/hardness.py`: build a `ProblemSpec` and
   call `problems.registry.register(spec)`. Take `checkpoint_metric` and
   `checkpoint_order` from your objective class so the metric name has a
   single home.

A minimal working example is `tests/test_toy_problem_extension.py`, which
builds an example problem out of exactly these pieces from test code alone. To turn it into a real problem, follow its shape with
your own backend and environment, move the module under `src/problems/`, and
register it from `problems/__init__.py`.

Before you call it done:

- Model files go in `src/env/<problem>/` and input data in
  `data/<problem>/`; both are git-ignored, so add `PUT_*_HERE.txt`
  placeholders.
- Hyperparameter defaults follow the existing pattern: `os.getenv` fallbacks
  inside `add_env_args`, documented in `.env`.
- Tests swap the backend for a stub, so they need no model file or GPU; see
  `tests/conftest.py` and the existing environment tests.
- `uv run pytest`, `uv run ruff check .`, and `uv run mypy .` all pass.

## Local model inference

The hardness prediction model also works on its own, with no reinforcement
learning involved. This is useful when you just want predictions: run a CSV
of candidate compositions through the model and get predicted hardness values
back.

You need a trained model package, a ZIP file with this layout:

```text
training_config.json
features.json
committee_scalers.pkl
committee_models/
```

### Command line

The simplest way. Point the script at the model ZIP, your input CSV, and
where to write the results:

```bash
uv run python scripts/run_model_inference.py \
  --model src/env/hardness/XGB_model_selection_package.zip \
  --input data/hardness/input.csv \
  --output data/hardness/output.csv
```

### Python API

From your own script or notebook:

```python
import pandas as pd

from env import InferenceModel

model = InferenceModel("src/env/hardness/XGB_model_selection_package.zip")

input_df = pd.DataFrame(
    [
        {
            "Structure": "BCC",
            "frac_Al": 0.2,
            "frac_Cr": 0.1,
        }
    ]
)

result_df = model.predict(input_df)
print(result_df)
```

The result has a `Predicted <target>` column and an `Uncertainty <target>`
column.

There is also a one-call shortcut that reads a CSV directly:

```python
from env import predict

result_df = predict(
    model_package_path="src/env/hardness/XGB_model_selection_package.zip",
    input_data="data/hardness/input.csv",
)
```

### Categorical columns

Give the model the same columns the training data had, in their original
form. A column like `Structure` with values such as `"BCC"` works as is. If
the training run encoded such columns (one-hot and the like), the package
reads that from its own config and applies the same encoding itself. You
never have to build columns like `Structure_FCC` by hand.

## Tests, lint, and types

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

Always go through `uv run`. A globally installed mypy cannot see the project's
virtualenv packages and reports false errors. The test suite stubs the
inference model, so it runs without the model ZIP or a GPU.
