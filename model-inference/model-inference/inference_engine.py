import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import load_data
from utils import committee, file_process


class _InferenceConfig:
    """Minimal config object required by committee.test_committee_regressor."""

    def __init__(self, model_type, targets):
        self.regressors = {"type": model_type}
        self.TARGETS = list(targets)
        self.y_col = list(targets)


def _prediction_columns(targets, output_count, prefix):
    if output_count == len(targets):
        return [f"{prefix} {target}" for target in targets]
    return [f"{prefix} {idx + 1}" for idx in range(output_count)]


def _make_unique_columns(df):
    df = df.copy()
    seen = {}
    unique_columns = []
    for column in df.columns:
        column = str(column)
        if column in seen:
            seen[column] += 1
            unique_columns.append(f"{column}_{seen[column]}")
        else:
            seen[column] = 0
            unique_columns.append(column)
    df.columns = unique_columns
    return df


def _to_dataframe(input_data):
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    if isinstance(input_data, (str, os.PathLike)):
        path = str(input_data)
        df, _ = load_data.read_table_flexible(path)
        return df
    if isinstance(input_data, dict):
        return pd.DataFrame([input_data])
    if isinstance(input_data, list):
        return pd.DataFrame(input_data)
    raise TypeError(
        "input_data must be a pandas DataFrame, dict, list of dicts, or CSV/Excel file path."
    )


class InferenceModel:
    """Load a trained platform model package and run local inference.

    Parameters
    ----------
    model_package_path:
        ZIP downloaded from the platform train valid or model selection page.
    extract_dir:
        Optional persistent extraction directory. If omitted, a temporary directory
        is created for the lifetime of this InferenceModel instance.
    """

    def __init__(self, model_package_path, extract_dir=None):
        self.model_package_path = Path(model_package_path)
        if not self.model_package_path.exists():
            raise FileNotFoundError(f"Model package not found: {self.model_package_path}")

        self._temporary_dir = None
        if extract_dir is None:
            self._temporary_dir = tempfile.TemporaryDirectory(prefix="model_inference_")
            self.extract_dir = Path(self._temporary_dir.name)
        else:
            self.extract_dir = Path(extract_dir)
            if self.extract_dir.exists():
                shutil.rmtree(self.extract_dir)
            self.extract_dir.mkdir(parents=True, exist_ok=True)

        file_process.unzip_strip_top_level(str(self.model_package_path), str(self.extract_dir))
        self.package = file_process.load_model_package(str(self.extract_dir))
        self.metadata = self.package["metadata"]
        self.model_type = self.package["model_type"]

        features_block = self.metadata.get("features", {})
        self.model_features = list(features_block.get("model_features") or features_block.get("features") or [])
        self.original_features = list(features_block.get("original_features") or self.model_features)
        self.targets = list(features_block.get("target") or [])
        if not self.model_features:
            raise ValueError("No model features were found in training_config.json.")
        if not self.targets:
            raise ValueError("No target columns were found in training_config.json.")

        self.models = committee.load_models_from_folder(self.package["model_folder"])
        self.scalers = committee.load_scalers(self.package["model_folder"])
        self.config = _InferenceConfig(self.model_type, self.targets)

    def prepare_features(self, input_data):
        raw_df = _to_dataframe(input_data)
        encoded_df = file_process.apply_package_feature_encoding(raw_df, self.metadata)
        missing_features = [feature for feature in self.model_features if feature not in encoded_df.columns]
        if missing_features:
            raise ValueError(
                "Missing required model features after preprocessing: "
                + ", ".join(missing_features)
            )

        x_df = encoded_df[self.model_features].copy()
        for column in x_df.columns:
            if x_df[column].dtype == bool:
                x_df[column] = x_df[column].astype(float)
            else:
                x_df[column] = pd.to_numeric(x_df[column], errors="coerce")
        x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return raw_df, x_df

    def predict(self, input_data, include_input=True):
        raw_df, x_df = self.prepare_features(input_data)
        y_pred, y_std, _, _ = committee.test_committee_regressor(
            self.config,
            self.models,
            self.scalers,
            x_df.to_numpy(),
        )

        pred_cols = _prediction_columns(self.targets, y_pred.shape[1], "Predicted")
        std_cols = _prediction_columns(self.targets, y_std.shape[1], "Uncertainty")
        pred_df = pd.DataFrame(y_pred, columns=pred_cols)
        std_df = pd.DataFrame(y_std, columns=std_cols)

        if include_input:
            result_df = pd.concat([raw_df.reset_index(drop=True), pred_df, std_df], axis=1)
        else:
            result_df = pd.concat([pred_df, std_df], axis=1)
        return _make_unique_columns(result_df)

    def predict_array(self, input_data):
        raw_df, x_df = self.prepare_features(input_data)
        y_pred, y_std, _, _ = committee.test_committee_regressor(
            self.config,
            self.models,
            self.scalers,
            x_df.to_numpy(),
        )
        return y_pred, y_std

    def close(self):
        if self._temporary_dir is not None:
            self._temporary_dir.cleanup()
            self._temporary_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def predict(model_package_path, input_data, include_input=True):
    with InferenceModel(model_package_path) as model:
        return model.predict(input_data, include_input=include_input)
