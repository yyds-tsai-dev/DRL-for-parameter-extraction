import numpy as np


def calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Percentage Error (RMSPE).
    RMSPE = sqrt(1/n * sum((y_true - y_pred)^2 / y_true^2))
    """
    y_true_safe = np.where(np.abs(y_true) < 1e-12, 1e-12, y_true)
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true_safe)))
    return float(rmspe)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE).
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return float(rmse)


def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Normalized Root Mean Squared Error (NRMSE).

    The normalization is done by dividing the RMSE by the Root Mean Square (RMS)
    of the true values (y_true). The result is returned as a percentage.

    NRMSE = (RMSE(y_true, y_pred) / RMS(y_true)) * 100
          = (sqrt(mean((y_true - y_pred)^2)) / sqrt(mean(y_true^2))) * 100
    """
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    rms_true = np.sqrt(np.mean(np.square(y_true)))

    if rms_true < 1e-12:
        return 1e4 if rmse > 1e-12 else 0.0

    nrmse = (rmse / rms_true) * 100
    return float(nrmse)
