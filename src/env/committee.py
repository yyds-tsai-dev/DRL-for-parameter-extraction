import os
import pickle

import joblib
import numpy as np
import pandas as pd

try:
    from tensorflow import keras
except ModuleNotFoundError:
    keras = None

SINGLE_MODEL_REGRESSORS = {"MLP", "RF", "SVR", "KNN", "GBR", "GPR"}
CLASSIFICATION_MODELS = {"GPC", "MLP_CLS"}
CLUSTERING_MODELS = {"KMeans", "Hierarchical"}
NON_REGRESSION_MODELS = CLASSIFICATION_MODELS | CLUSTERING_MODELS


def _require_keras():
    if keras is None:
        raise ModuleNotFoundError("TensorFlow is required for loading Keras model packages.")
    return keras




def load_scalers(folder_name):
    with open(folder_name + "/committee_scalers.pkl", "rb") as f:
        return pickle.load(f)


def load_models_from_folder(folder_path):
    folder_path = folder_path + "/committee_models"
    models = []

    def _sort_key(filename):
        digits = "".join(ch for ch in filename if ch.isdigit())
        return (int(digits) if digits else 10**9, filename)

    for filename in sorted(os.listdir(folder_path), key=_sort_key):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".keras"):
            model = _require_keras().models.load_model(filepath)
            models.append(model)
        elif filename.endswith(".pkl"):
            with open(filepath, "rb") as f:
                models = pickle.load(f)
        elif filename.endswith(".joblib"):
            model = joblib.load(filepath)
            models.append(model)
        elif os.path.isdir(filepath):
            try:
                model = _require_keras().models.load_model(filepath)
                models.append(model)
            except Exception as e:
                print(f"Can't import model: {filepath}, Error: {e}")
    return models




def test_committee_regressor(config, committee_models, scalers, data_x):
    in_scalers = scalers["in_scalers"]
    out_scalers = scalers["out_scalers"]
    model_type = config.regressors["type"]

    if model_type in NON_REGRESSION_MODELS:
        predictions = []
        num_models = min(len(committee_models), len(in_scalers))
        for model_idx in range(num_models):
            model = committee_models[model_idx]
            x_scaled = in_scalers[model_idx].transform(data_x)
            if model_type == "Hierarchical":
                pred = model.fit_predict(x_scaled)
            else:
                pred = model.predict(x_scaled)
            predictions.append(np.asarray(pred).reshape(-1))
        all_predictions = np.asarray(predictions).T
        if model_type in CLUSTERING_MODELS:
            means = all_predictions[:, 0].reshape(-1, 1)
        else:
            voted = []
            for row in all_predictions:
                voted.append(pd.Series(row).mode().iloc[0])
            means = np.asarray(voted).reshape(-1, 1)
        numeric_predictions = pd.DataFrame(all_predictions).apply(pd.to_numeric, errors="coerce").to_numpy()
        if np.isnan(numeric_predictions).all():
            stds = np.zeros_like(means, dtype=float)
        else:
            stds = np.nanstd(numeric_predictions, axis=1).reshape(-1, 1)
        return means, stds, None, all_predictions

    if model_type in SINGLE_MODEL_REGRESSORS:
        num_outputs = len(config.TARGETS)
        num_models = min(len(committee_models), len(in_scalers), len(out_scalers))
        if num_models == 0:
            raise ValueError("No complete model/scaler pairs were found in the model package.")
    else:
        num_outputs = len(committee_models)
        num_models = len(committee_models[0])

    all_predictions = []

    if model_type == "MLP":
        preds_per_output = []
        for model_idx in range(num_models):
            model = committee_models[model_idx]
            x_scaled = in_scalers[model_idx].transform(data_x)
            pred = model(x_scaled)
            pred = out_scalers[model_idx].inverse_transform(pred)
            preds_per_output.append(pred.flatten())
        all_predictions.append(preds_per_output)

    elif model_type in SINGLE_MODEL_REGRESSORS:
        preds_per_output = []
        for model_idx in range(num_models):
            model = committee_models[model_idx]
            x_scaled = in_scalers[model_idx].transform(data_x)
            pred = model.predict(x_scaled)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            pred = out_scalers[model_idx].inverse_transform(pred)
            preds_per_output.append(pred.flatten())
        all_predictions.append(preds_per_output)

    else:
        for output_idx in range(num_outputs):
            preds_per_output = []
            for model_idx in range(num_models):
                model = committee_models[output_idx][model_idx]
                x_scaled = in_scalers[output_idx][model_idx].transform(data_x)
                pred = model.predict(x_scaled)
                pred = out_scalers[output_idx][model_idx].inverse_transform(pred.reshape(-1, 1))
                preds_per_output.append(pred.flatten())
            all_predictions.append(preds_per_output)

    all_predictions = np.array(all_predictions)
    all_predictions = np.transpose(all_predictions, (0, 2, 1))
    means = all_predictions.mean(axis=2)
    stds = all_predictions.std(axis=2)

    if model_type in SINGLE_MODEL_REGRESSORS:
        out_shape = pred.shape
        means = np.reshape(means, out_shape)
        stds = np.reshape(stds, out_shape)
        per_output_covs = np.cov(all_predictions[0])
    else:
        means = means.T
        stds = stds.T
        per_output_covs = []
        for i in range(num_outputs):
            per_output_covs.append(np.cov(all_predictions[i]))

    return means, stds, per_output_covs, all_predictions
