# Model-Inference

Standalone local inference package for models exported from the machine learning platform.

This folder is designed for collaborators who want to embed a trained platform model in a local Python program, reinforcement learning loop, optimization workflow, or batch inference script. Streamlit is not required.

## 1. Install

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Add A Model Package

Copy the ZIP downloaded from the platform train valid or model selection page into:

```text
model_packages/
```

The ZIP must contain:

```text
training_config.json
features.json
committee_scalers.pkl
committee_models/
```

## 3. Python API Usage

```python
import pandas as pd
from inference_engine import InferenceModel

model = InferenceModel("model_packages/checkpoint.zip")

input_df = pd.DataFrame([
    {
        "Structure": "BCC",
        "frac_Al": 0.2,
        "frac_Cr": 0.1,
    }
])

result_df = model.predict(input_df)
print(result_df)
```

The result contains prediction and uncertainty columns:

```text
Predicted <target>
Uncertainty <target>
```

## 4. Function-Style Usage

```python
from inference_engine import predict

result_df = predict(
    model_package_path="model_packages/checkpoint.zip",
    input_data="sample_data/input.csv",
)
```

## 5. Command Line Usage

```bash
python example_run.py --model model_packages/checkpoint.zip --input sample_data/input.csv --output outputs/predictions.csv
```

## 6. Reinforcement Learning Loop Example

```python
from inference_engine import InferenceModel

model = InferenceModel("model_packages/checkpoint.zip")

def reward_from_state(state_dict):
    result = model.predict(state_dict, include_input=False)
    predicted_value = result.iloc[0]["Predicted target"]
    uncertainty = result.iloc[0]["Uncertainty target"]
    return predicted_value - 0.1 * uncertainty
```

Replace `"Predicted target"` and `"Uncertainty target"` with the actual target column names shown in the output.

## 7. Categorical Features

Use raw feature columns in the same form as training data. If the original training workflow used one-hot encoding, label encoding, or target encoding, this package reads that information from `training_config.json` and applies it automatically.

Example:

```python
input_df = pd.DataFrame([{"Structure": "FCC", "frac_Ni": 0.25}])
```

You do not need to manually create columns like `Structure_FCC`; the package will align features to the model package.

## 8. Notes

- This package is for inference only. It does not train models.
- Keep the model ZIP and this folder together when sharing with collaborators.
- If TensorFlow prints GPU or CPU messages during import, that is normal when loading Keras-capable inference utilities.
