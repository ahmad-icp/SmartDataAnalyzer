"""Evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error


def evaluate_predictions(task_type: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Return evaluation payload by task type."""
    if task_type == "classification":
        return {
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
    }
