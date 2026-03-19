"""Local intelligence engine for automated dataset insights and recommendations."""

from __future__ import annotations

import pandas as pd

from utils.helpers import infer_task_type


def detect_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return numerical and categorical feature lists."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(exclude="number").columns.tolist()
    return {"numeric": numeric, "categorical": categorical}


def detect_high_correlation(df: pd.DataFrame, threshold: float = 0.85) -> list[tuple[str, str, float]]:
    """Detect highly correlated numeric feature pairs."""
    corr = df.select_dtypes(include="number").corr().abs()
    pairs: list[tuple[str, str, float]] = []
    if corr.empty:
        return pairs
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1 :]:
            value = float(corr.loc[col_a, col_b])
            if value >= threshold:
                pairs.append((col_a, col_b, value))
    return pairs


def suggest_preprocessing_steps(df: pd.DataFrame) -> list[str]:
    """Generate rule-based preprocessing suggestions."""
    suggestions: list[str] = []
    missing_cells = int(df.isna().sum().sum())
    if missing_cells > 0:
        suggestions.append("Dataset contains missing values. Consider imputation.")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        suggestions.append(f"Dataset contains {duplicate_rows} duplicate rows. Consider removal.")

    types = detect_feature_types(df)
    if types["categorical"]:
        suggestions.append("Categorical features detected. Consider encoding before modeling.")
    if types["numeric"]:
        suggestions.append("Numeric features detected. Consider scaling for linear models.")

    high_corr = detect_high_correlation(df)
    if high_corr:
        suggestions.append("High correlation detected. Consider feature reduction.")

    if not suggestions:
        suggestions.append("Dataset appears clean. Proceed to feature engineering and modeling.")
    return suggestions


def suggest_model_type(df: pd.DataFrame, target_column: str) -> str:
    """Suggest modeling task type from target variable."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    task_type = infer_task_type(df[target_column])
    if task_type == "classification":
        return "Target variable appears categorical or low-cardinality → use classification models."
    return "Target variable appears continuous → use regression models."


def generate_intelligence_report(df: pd.DataFrame, target_column: str | None = None) -> list[str]:
    """Return comprehensive local intelligence report."""
    insights = suggest_preprocessing_steps(df)
    if target_column:
        insights.append(suggest_model_type(df, target_column))
    return insights
