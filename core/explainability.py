"""Explainability module with SHAP integration."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_shap_summary(model: object, X_sample: pd.DataFrame) -> dict[str, Any]:
    """Compute global SHAP values summary for a sample."""
    import shap

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    mean_abs = abs(shap_values.values).mean(axis=0)
    importance = pd.DataFrame({"feature": X_sample.columns, "importance": mean_abs})
    importance = importance.sort_values("importance", ascending=False)
    return {
        "feature_importance": importance,
        "shap_values": shap_values,
    }


def explain_single_prediction(model: object, row: pd.DataFrame) -> dict[str, Any]:
    """Compute local SHAP explanation for one row."""
    import shap

    explainer = shap.Explainer(model, row)
    values = explainer(row)
    return {
        "base_value": values.base_values[0] if hasattr(values, "base_values") else None,
        "contributions": values.values[0].tolist(),
    }
