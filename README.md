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
ZIP in `env/hardness/` and input CSVs in `data/hardness/`; the `PUT_*_HERE.txt`
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

## Tests, lint, and types

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

Always go through `uv run`. A globally installed mypy cannot see the project's
virtualenv packages and reports false errors. The test suite stubs the
inference model, so it runs without the model ZIP or a GPU.

## Adding a new problem

The harness resolves `--env <name>` through `problems/registry.py`, and
everything problem-specific travels in a `ProblemSpec`. Adding a problem takes
no edits to shared harness code. The executable proof of that claim is
`tests/test_toy_problem_extension.py`, which registers a complete toy problem
from test code alone; read it alongside this section.

A problem consists of five pieces:

1. A prediction backend, which turns one candidate into a predicted value.
   Implement the `PredictionBackend` protocol in `env/backends.py`:
   `predict(features) -> PredictionResult` plus `close()`. Committee
   model-package ZIPs can reuse `CommitteePackageBackend` as-is. An ANN or
   other surrogate implements the protocol directly and does not need the
   committee ZIP format or the verilogae toolchain. Uncertainties are
   diagnostic only; keep them out of the reward.
2. An objective, which turns a predicted value into reward and success. Reuse
   `ThresholdMaximizeObjective` or `NRMSEMinimizeObjective` from
   `env/objectives.py` when the semantics match, or add a class with the same
   shape: `RANKED_METRIC`, `RANKED_ORDER`, and the reward and success methods.
   Episode control (termination, truncation) belongs to your environment, not
   the objective. ADR 0003 records why.
3. A `gymnasium.Env` whose observations stay inside finite-bound float32 Boxes
   in every reset mode. Accept your backend through env-config injection so
   tests can stub it. `MaterialHardnessEnv` (`prediction_backend_cls`) and
   `EEHEMTEnv_Measure_VDS` (`simulator_factory`) show the two established
   patterns.
4. A training module exposing `add_env_args(parser, current_dir)`,
   `build_env_config(args)`, `build_ppo_config(args, *, num_learners,
   num_gpus_per_learner)` (delegate to
   `training.ppo_common.build_base_ppo_config`), `build_checkpoint_config()`,
   and a `<NAME>_WANDB_PROJECT` constant. `training/hardness_ppo.py` is the
   reference implementation, at roughly 90 lines.
5. Registration: build a `ProblemSpec` and call
   `problems.registry.register(spec)`, following `problems/hardness.py`. Take
   `checkpoint_metric` and `checkpoint_order` from your objective class so the
   metric name keeps a single home.

Before you call it done:

- Committee ZIPs go in `env/<problem>/`, input data in `data/<problem>/`. Both
  are git-ignored; add `PUT_*_HERE.txt` placeholders.
- Hyperparameter defaults follow the existing pattern: `os.getenv` fallbacks
  inside `add_env_args`, documented in `.env`.
- Tests stub the backend through your injection seam. No model artifact or GPU
  needed; see `tests/conftest.py` and the existing env tests.
- `uv run pytest && uv run ruff check . && uv run mypy .` all green.
- Record new layer-boundary decisions in `docs/adr/` and
  `.codebase-memory/adr.md`.

To promote the toy example into a real problem, copy the shape of
`tests/test_toy_problem_extension.py`, swap in your backend and env, move the
module under `problems/`, and register it from `problems/__init__.py`.

## Local model inference

The hardness prediction stack under `env/` also works standalone, outside any
RL loop. It loads trained platform model packages for local scripts, batch
inference, or other optimization workflows.

The ZIP package must contain:

```text
training_config.json
features.json
committee_scalers.pkl
committee_models/
```

### Python API

```python
import pandas as pd

from env import InferenceModel

model = InferenceModel("env/hardness/XGB_model_selection_package.zip")

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

The result contains `Predicted <target>` and `Uncertainty <target>` columns.
For a typed interface without DataFrame column names, wrap the package in
`env.backends.CommitteePackageBackend` and call `predict(features)` to get a
`PredictionResult`.

### Function API

```python
from env import predict

result_df = predict(
    model_package_path="env/hardness/XGB_model_selection_package.zip",
    input_data="data/hardness/input.csv",
)
```

### Command line

```bash
uv run python scripts/run_model_inference.py \
  --model env/hardness/XGB_model_selection_package.zip \
  --input data/hardness/input.csv \
  --output data/hardness/output.csv
```

### Categorical features

Use raw feature columns in the same form as the training data. If the original
training workflow used one-hot, label, or target encoding, the package reads
that from `training_config.json` and applies it automatically. You do not need
to create columns like `Structure_FCC` by hand.

## Where things are documented

- `CONTEXT.md` defines the project's shared vocabulary (I-V Curve, Curve
  Condition, Feasible Material Composition, Problem Spec, and so on). Use
  these terms in code and discussion.
- `docs/adr/` holds architecture decision records. ADR 0001 locks the EEHEMT
  NRMSE objective, ADR 0002 the IR-drop solver strategy, ADR 0003 the problem
  registry, backends, and objectives.
- `docs/how-to-add-a-problem.md` is the standalone version of the extension
  guide above.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` keep the design
  documents and implementation plans that produced the current architecture.
