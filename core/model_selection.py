"""AutoML-lite model selection with basic overfitting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from utils.helpers import infer_task_type


@dataclass
class ModelSelectionResult:
    """Results payload for trained model candidates."""

    task_type: str
    metric_name: str
    best_model_name: str
    best_model: Any
    performance_table: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    best_train_score: float
    best_test_score: float


def run_model_selection(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
) -> ModelSelectionResult:
    """Train multiple candidate models and select the best model by task metric."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    if len(df) < 5:
        raise ValueError("At least 5 rows are required for model selection.")

    X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=False)
    y = df[target_column]

    task_type = infer_task_type(y)
    stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        }
        scorer = accuracy_score
        metric_name = "accuracy"
        greater_is_better = True
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "Bayesian Ridge": BayesianRidge(),
        }
        scorer = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))
        metric_name = "rmse"
        greater_is_better = False

    records: list[dict[str, float | str]] = []
    fitted: dict[str, tuple[Any, np.ndarray, np.ndarray, float, float]] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_score = float(scorer(y_train, train_pred))
        test_score = float(scorer(y_test, test_pred))

        records.append({"model": name, f"train_{metric_name}": train_score, metric_name: test_score})
        fitted[name] = (model, test_pred, train_pred, train_score, test_score)

    perf = pd.DataFrame(records).sort_values(metric_name, ascending=not greater_is_better).reset_index(drop=True)
    best_name = str(perf.loc[0, "model"])
    best_model, y_pred, _train_pred, train_score, test_score = fitted[best_name]

    return ModelSelectionResult(
        task_type=task_type,
        metric_name=metric_name,
        best_model_name=best_name,
        best_model=best_model,
        performance_table=perf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        best_train_score=float(train_score),
        best_test_score=float(test_score),
    )
