"""Evaluation metrics for regression output."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute a standard suite of regression metrics.

    Returns:
        {
          "rmse": float,   # Root Mean Squared Error
          "mae":  float,   # Mean Absolute Error
          "r2":   float,   # Coefficient of Determination
          "mape": float,   # Mean Absolute Percentage Error (%)
        }
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE — guard against zero targets
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}
