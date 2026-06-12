import os
import pickle
import multiprocessing
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:
    tf = None
    keras = None

SINGLE_MODEL_REGRESSORS = {"MLP", "RF", "SVR", "KNN", "GBR", "GPR"}
XGB_REGRESSORS = {"XGB"}
CLASSIFICATION_MODELS = {"GPC", "MLP_CLS"}
CLUSTERING_MODELS = {"KMeans", "Hierarchical"}
NON_REGRESSION_MODELS = CLASSIFICATION_MODELS | CLUSTERING_MODELS


def _require_tensorflow():
    if tf is None:
        raise ModuleNotFoundError("TensorFlow is required for Keras/MLP model operations.")
    return tf


def _require_keras():
    if keras is None:
        raise ModuleNotFoundError("TensorFlow is required for loading Keras model packages.")
    return keras


def make_input_scaler(config):
    """Create the input scaler selected in the training page."""
    method = getattr(config, "normalization_method", "z-score")
    method = (method or "z-score").lower()

    if method in {"none", "raw"}:
        return FunctionTransformer(validate=False)
    if method in {"z-score", "zscore", "standard"}:
        return StandardScaler()
    if method in {"min-max", "minmax"}:
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()
    if method in {"log", "log1p"}:
        return FunctionTransformer(np.log1p, validate=False)
    raise ValueError(f"Unsupported normalization method: {method}")


def train_committee_Regressor_v2(model, config, samples=64):
    """Train a committee of models with bootstrap sampling."""
    scalers = {}

    in_scaler = make_input_scaler(config)
    out_scaler = StandardScaler()

    models = []
    in_scalers = []
    out_scalers = []

    print(config.data_x.shape, config.data_y.shape, len(config.df_idx))

    for _ in range(samples):
        shuffle_idx = config.df_idx.sample(frac=1.0, replace=True)[0]
        shuffle_idx = list(shuffle_idx)

        x_boot = in_scaler.fit_transform(config.data_x[shuffle_idx])
        y_boot = out_scaler.fit_transform(config.data_y[shuffle_idx])
        if y_boot.shape[-1] == 1:
            y_boot = y_boot.ravel()

        if config.regressors["type"] == "MLP":
            history = model.fit(
                x_boot,
                y_boot,
                batch_size=config.batch_size,
                epochs=config.epochs,
                verbose=config.verbose,
                callbacks=config.callbacks,
                validation_split=config.validation_split,
                validation_data=config.validation_data,
                shuffle=config.shuffle,
                class_weight=config.class_weight,
                sample_weight=config.sample_weight,
                initial_epoch=config.initial_epoch,
                steps_per_epoch=config.steps_per_epoch,
                validation_steps=config.validation_steps,
                validation_batch_size=config.validation_batch_size,
                validation_freq=config.validation_freq,
            )
        elif config.regressors["type"] == "RF":
            model.fit(x_boot, y_boot)
        elif config.regressors["type"] == "XGB":
            X_train, X_val, y_train, y_val = train_test_split(
                x_boot, y_boot, test_size=0.2, random_state=42, stratify=None
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=True,
            )
            evals_result = model.evals_result()

        models.append(model)
        in_scalers.append(deepcopy(in_scaler))
        out_scalers.append(deepcopy(out_scaler))

    scalers["in_scalers"] = in_scalers
    scalers["out_scalers"] = out_scalers
    if config.regressors["type"] == "MLP":
        return models, scalers, history
    if config.regressors["type"] == "RF":
        return models, scalers
    return models, scalers, evals_result


def train_committee2_parallel(model_template, config, samples=64, n_jobs=-1):
    """Train a committee with controlled parallelism."""
    scalers = {}
    n_outputs = 0 if config.data_y is None else config.data_y.shape[1]
    model_type = config.regressors["type"]

    def _raise_if_stop_requested():
        stop_flag_file = getattr(config, "stop_flag_file", None)
        if stop_flag_file and os.path.exists(stop_flag_file):
            raise RuntimeError("Training stopped by user.")

    def _safe_cpu_count():
        return max(1, multiprocessing.cpu_count())

    def _resolve_n_jobs(requested_n_jobs):
        if requested_n_jobs in (None, 0):
            requested_n_jobs = 1
        if requested_n_jobs == -1:
            requested_n_jobs = _safe_cpu_count()
        requested_n_jobs = max(1, int(requested_n_jobs))
        requested_n_jobs = min(requested_n_jobs, max(1, samples))

        # Keras/GPR/GPC + joblib workers on Windows can easily exhaust memory/CPU.
        if model_type in {"MLP", "MLP_CLS", "GPR", "GPC", "Hierarchical"}:
            return 1
        return requested_n_jobs

    def _configure_tf_threads():
        tensorflow = _require_tensorflow()
        intra_threads = getattr(config, "tf_intra_op_threads", None)
        inter_threads = getattr(config, "tf_inter_op_threads", None)
        try:
            if intra_threads is not None:
                tensorflow.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
            if inter_threads is not None:
                tensorflow.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
        except RuntimeError:
            pass

    resolved_n_jobs = _resolve_n_jobs(getattr(config, "committee_n_jobs", n_jobs))
    if model_type == "MLP":
        _configure_tf_threads()

    def _train_single_bootstrap_model(sample_seed):
        _raise_if_stop_requested()
        rng = np.random.default_rng(seed=sample_seed)
        bootstrap_idx = rng.integers(0, len(config.data_x), size=len(config.data_x))

        if model_type in SINGLE_MODEL_REGRESSORS:
            in_scaler = make_input_scaler(config)
            out_scaler = StandardScaler()

            x_boot = in_scaler.fit_transform(config.data_x[bootstrap_idx])
            y_boot = out_scaler.fit_transform(config.data_y[bootstrap_idx])

            model = deepcopy(model_template)
            history = (
                model.fit(
                    x_boot,
                    y_boot,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    verbose=config.verbose,
                    callbacks=config.callbacks,
                    validation_split=config.validation_split,
                    validation_data=config.validation_data,
                    shuffle=config.shuffle,
                    class_weight=config.class_weight,
                    sample_weight=config.sample_weight,
                    initial_epoch=config.initial_epoch,
                    steps_per_epoch=config.steps_per_epoch,
                    validation_steps=config.validation_steps,
                    validation_batch_size=config.validation_batch_size,
                    validation_freq=config.validation_freq,
                )
                if model_type == "MLP"
                else model.fit(x_boot, y_boot)
            )
            return model, deepcopy(in_scaler), deepcopy(out_scaler), deepcopy(history) if model_type == "MLP" else None

        if model_type in NON_REGRESSION_MODELS:
            in_scaler = make_input_scaler(config)
            x_boot = in_scaler.fit_transform(config.data_x[bootstrap_idx])
            model = deepcopy(model_template)
            if model_type in CLASSIFICATION_MODELS:
                y_boot = config.data_y[bootstrap_idx].ravel()
                model.fit(x_boot, y_boot)
            else:
                model.fit(x_boot)
            return model, deepcopy(in_scaler), None, None

        models = []
        in_scalers = []
        out_scalers = []
        evals_results = []

        for output_idx in range(n_outputs):
            _raise_if_stop_requested()
            in_scaler = make_input_scaler(config)
            out_scaler = StandardScaler()

            x_boot = in_scaler.fit_transform(config.data_x[bootstrap_idx])
            y_boot = out_scaler.fit_transform(config.data_y[bootstrap_idx, output_idx].reshape(-1, 1))

            X_train, X_val, y_train, y_val = train_test_split(
                x_boot, y_boot, test_size=0.2, random_state=42
            )
            model = deepcopy(model_template)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=True,
            )

            models.append(model)
            in_scalers.append(deepcopy(in_scaler))
            out_scalers.append(deepcopy(out_scaler))
            evals_results.append(model.evals_result())

        return models, in_scalers, out_scalers, evals_results

    if resolved_n_jobs == 1:
        results = [_train_single_bootstrap_model(seed) for seed in range(samples)]
    else:
        results = Parallel(n_jobs=resolved_n_jobs, verbose=10)(
            delayed(_train_single_bootstrap_model)(seed) for seed in range(samples)
        )

    if model_type in SINGLE_MODEL_REGRESSORS | NON_REGRESSION_MODELS:
        models = [res[0] for res in results]
        scalers["in_scalers"] = [res[1] for res in results]
        scalers["out_scalers"] = [res[2] for res in results]
        histories = [res[3] for res in results]
        print(f"Finished training {len(models)} committee members with n_jobs={resolved_n_jobs}.")
        return models, scalers, histories[-1] if model_type == "MLP" else None

    models = list(map(list, zip(*[res[0] for res in results])))
    evals_results = list(map(list, zip(*[res[3] for res in results])))
    scalers["in_scalers"] = [res[1] for res in results]
    scalers["out_scalers"] = [res[2] for res in results]
    scalers["in_scalers"] = list(map(list, zip(*scalers["in_scalers"])))
    scalers["out_scalers"] = list(map(list, zip(*scalers["out_scalers"])))
    print(f"Finished training {len(models[0])} committee members per output with n_jobs={resolved_n_jobs}.")
    return models, scalers, evals_results


def train_committee_Regressor(model, config, samples=64):
    """Train a committee of the sample ML model with bootstrap."""
    scalers = {}
    n_outputs = config.data_y.shape[1]

    if config.regressors["type"] in {"MLP", "RF"}:
        in_scaler = make_input_scaler(config)
        out_scaler = StandardScaler()

        committee_models = []
        in_scalers = []
        out_scalers = []

        print(config.data_x.shape, config.data_y.shape, len(config.df_idx))

        for _ in range(samples):
            shuffle_idx = config.df_idx.sample(frac=1.0, replace=True)[0]
            shuffle_idx = list(shuffle_idx)

            x_boot = in_scaler.fit_transform(config.data_x[shuffle_idx])
            y_boot = out_scaler.fit_transform(config.data_y[shuffle_idx])
            if y_boot.shape[-1] == 1:
                y_boot = y_boot.ravel()

            if config.regressors["type"] == "MLP":
                history = model.fit(
                    x_boot,
                    y_boot,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    verbose=config.verbose,
                    callbacks=config.callbacks,
                    validation_split=config.validation_split,
                    validation_data=config.validation_data,
                    shuffle=config.shuffle,
                    class_weight=config.class_weight,
                    sample_weight=config.sample_weight,
                    initial_epoch=config.initial_epoch,
                    steps_per_epoch=config.steps_per_epoch,
                    validation_steps=config.validation_steps,
                    validation_batch_size=config.validation_batch_size,
                    validation_freq=config.validation_freq,
                )
                committee_models.append(model)
            else:
                model.fit(x_boot, y_boot)
                committee_models.append(model)

            in_scalers.append(deepcopy(in_scaler))
            out_scalers.append(deepcopy(out_scaler))

    else:
        committee_models = [[] for _ in range(n_outputs)]
        in_scalers = [[] for _ in range(n_outputs)]
        out_scalers = [[] for _ in range(n_outputs)]
        evals_result = [[] for _ in range(n_outputs)]

        for output_idx in range(n_outputs):
            y_target = config.data_y[:, output_idx]
            for _ in range(samples):
                in_scaler = make_input_scaler(config)
                out_scaler = StandardScaler()
                shuffle_idx = config.df_idx.sample(frac=1.0, replace=True)[0]
                shuffle_idx = list(shuffle_idx)

                x_boot = in_scaler.fit_transform(config.data_x[shuffle_idx])
                y_boot = out_scaler.fit_transform(y_target[shuffle_idx].reshape(-1, 1))
                if y_boot.shape[-1] == 1:
                    y_boot = y_boot.ravel()

                X_train, X_val, y_train, y_val = train_test_split(
                    x_boot, y_boot, test_size=0.2, random_state=42
                )

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=True,
                )
                committee_models[output_idx].append(model)
                evals_result[output_idx].append(model.evals_result())
                in_scalers[output_idx].append(in_scaler)
                out_scalers[output_idx].append(out_scaler)

    scalers["in_scalers"] = in_scalers
    scalers["out_scalers"] = out_scalers
    if config.regressors["type"] == "MLP":
        return committee_models, scalers, history
    if config.regressors["type"] == "RF":
        return committee_models, scalers
    return committee_models, scalers, evals_result


def save_pickle(folder_name, committee_models):
    models_folder = folder_name + "/committee_models"
    if os.path.exists(models_folder):
        import shutil
        shutil.rmtree(models_folder)
    os.makedirs(models_folder)
    with open(folder_name + "/committee_models/committee_model.pkl", "wb") as f:
        pickle.dump(committee_models, f)


def save_joblib(folder_name, committee_models):
    models_folder = folder_name + "/committee_models"
    if os.path.exists(models_folder):
        import shutil
        shutil.rmtree(models_folder)
    os.makedirs(models_folder)
    for i in range(len(committee_models)):
        with open(folder_name + f"/committee_models/committee_model_{i}.joblib", "wb") as f:
            joblib.dump(committee_models[i], f)


def save_keras(folder_name, committee_models):
    models_folder = folder_name + "/committee_models"
    if os.path.exists(models_folder):
        import shutil
        shutil.rmtree(models_folder)
    os.makedirs(models_folder)
    for i in range(len(committee_models)):
        committee_models[i].save(folder_name + f"/committee_models/committee_model_{i}.keras")


def save_scalers(folder_name, scalers):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    with open(folder_name + "/committee_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)


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


def test_committee_regressor_v2(config, committee_models, scalers, data_x):
    models = committee_models
    in_scal = scalers["in_scalers"]
    out_scal = scalers["out_scalers"]

    predictions = []
    for idx, m in enumerate(models):
        if config.regressors["type"] == "MLP":
            pred = m(in_scal[idx].transform(data_x))
            pred = out_scal[idx].inverse_transform(pred)
        else:
            pred = m.predict(in_scal[idx].transform(data_x))
            pred = out_scal[idx].inverse_transform(pred.reshape(-1, 1))
        predictions.append(pred.flatten())

    predictions = np.asarray(predictions).T
    means = predictions.mean(axis=1)
    stds = predictions.std(axis=1)

    out_shape = pred.shape
    means = np.reshape(means, out_shape)
    stds = np.reshape(stds, out_shape)
    return means, stds, np.cov(predictions), predictions


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


def calc_stats(y, y_pred, stats={"mae", "r2", "pearson"}):
    y = pd.to_numeric(y.flatten(), errors="coerce")
    y_pred = np.asarray(y_pred).flatten()
    y = np.array(y)

    valid_mask = ~np.isnan(y) & ~np.isnan(y_pred)
    y = y[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays after NaN filtering. Check input data.")
    if np.any(np.isnan(y)) or np.any(np.isnan(y_pred)):
        raise ValueError("NaN values detected in y or y_pred")
    if np.any(np.isinf(y)) or np.any(np.isinf(y_pred)):
        raise ValueError("Infinite values detected in y or y_pred")

    stats_dict = {}
    if "mae" in stats:
        stats_dict["mae"] = mean_absolute_error(y, y_pred)
    if "mse" in stats:
        stats_dict["mse"] = mean_squared_error(y, y_pred)
    if "r2" in stats:
        stats_dict["r2"] = r2_score(y, y_pred)
    if "pearson" in stats:
        stats_dict["pearson"] = pearsonr(y, y_pred)[0]
    return stats_dict


def mean_absolute_error(y_true, y_pred):
    return sum(abs(y_true - y_pred)) / len(y_true)


def mean_squared_error(y_true, y_pred):
    return sum((y_true - y_pred) ** 2) / len(y_true)


def r2_score(y_true, y_pred):
    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - y_true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)
