import numpy as np

def calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Percentage Error (RMSPE).
    RMSPE = sqrt(1/n * sum((y_true - y_pred)^2 / y_true^2))
    """
    # Prevent division by zero for very small currents
    y_true_safe = np.where(np.abs(y_true) < 1e-12, 1e-12, y_true)
    rmspe = np.sqrt(np.mean(np.square((y_true - y_pred) / y_true_safe)))
    return rmspe

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Square Error (RMSE).
    RMSE = sqrt(mean((y_true - y_pred)^2))
    """    
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return rmse

def calculate_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Normalized Root Mean Squared Error (NRMSE).

    The normalization is done by dividing the RMSE by the Root Mean Square (RMS)
    of the true values (y_true). The result is returned as a percentage.

    NRMSE = (RMSE(y_true, y_pred) / RMS(y_true)) * 100
          = (sqrt(mean((y_true - y_pred)^2)) / sqrt(mean(y_true^2))) * 100
    """
    # Calculate the numerator: RMSE of the prediction and true values
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    # Calculate the denominator: RMS of the true values
    rms_true = np.sqrt(np.mean(np.square(y_true)))
    
    # Handle the edge case where the denominator might be zero.
    # If the true signal has zero energy (i.e., all zeros), NRMSE is undefined.
    # - If RMSE is also 0, the error is 0%.
    # - If RMSE > 0, the relative error is infinite.
    if rms_true < 1e-12:
        return 1e4 if rmse > 1e-12 else 0.0

    nrmse = (rmse / rms_true) * 100
    
    return nrmse