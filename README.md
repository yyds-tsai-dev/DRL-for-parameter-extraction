# DRL-on-parameter-extraction
Using DRL method accelerate EEHEMT parameter extraction for advanced CMOS technologies

## Setup

This project uses `uv` and keeps runtime dependencies in `pyproject.toml`.

```bash
uv sync
```

## Local Model Inference

The former standalone `model-inference` Python code is integrated directly under
`env/`, while the hardness model artifacts remain in `env/hardness/`. It can
load trained platform model packages for local Python programs,
reinforcement-learning loops, optimization workflows, or batch inference
scripts.

Model package artifacts are stored under:

```text
env/hardness/
```

The ZIP package must contain:

```text
training_config.json
features.json
committee_scalers.pkl
committee_models/
```

Sample hardness input/output files are stored under:

```text
data/hardness/
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

The result contains prediction and uncertainty columns:

```text
Predicted <target>
Uncertainty <target>
```

### Function API

```python
from env import predict

result_df = predict(
    model_package_path="env/hardness/XGB_model_selection_package.zip",
    input_data="data/hardness/Prof_yeh_250714_Trans_1_8_elementsd.csv",
)
```

### Command Line

```bash
uv run python scripts/run_model_inference.py \
  --model env/hardness/XGB_model_selection_package.zip \
  --input data/hardness/Prof_yeh_250714_Trans_1_8_elementsd.csv \
  --output data/hardness/Prof_yeh_250714_Trans_1_8_elementsd_inference.csv
```

### Reinforcement Learning Reward Example

```python
from env import InferenceModel

model = InferenceModel("env/hardness/XGB_model_selection_package.zip")


def reward_from_state(state_dict):
    result = model.predict(state_dict, include_input=False)
    predicted_value = result.iloc[0]["Predicted target"]
    uncertainty = result.iloc[0]["Uncertainty target"]
    return predicted_value - 0.1 * uncertainty
```

Replace `"Predicted target"` and `"Uncertainty target"` with the actual target
column names shown in the output.

### Categorical Features

Use raw feature columns in the same form as training data. If the original
training workflow used one-hot encoding, label encoding, or target encoding, the
package reads that information from `training_config.json` and applies it
automatically.

```python
input_df = pd.DataFrame([{"Structure": "FCC", "frac_Ni": 0.25}])
```

You do not need to manually create columns like `Structure_FCC`; the package
aligns features to the model package.
