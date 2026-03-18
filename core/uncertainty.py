"""Uncertainty estimation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge


def bayesian_prediction_interval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """Return mean prediction with standard deviation using Bayesian Ridge."""
    model = BayesianRidge()
    model.fit(X_train, y_train)
    mean_pred, std_pred = model.predict(X_test, return_std=True)
    return pd.DataFrame(
        {
            "prediction": mean_pred,
            "uncertainty": std_pred,
            "lower_95": mean_pred - 1.96 * std_pred,
            "upper_95": mean_pred + 1.96 * std_pred,
        }
    )
