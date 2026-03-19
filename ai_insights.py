"""Local, API-free insight generation helpers."""

from __future__ import annotations

import pandas as pd


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Generate simple heuristic insights without external APIs."""
    insights: list[str] = []
    if df.empty:
        return ["Dataset is empty."]

    missing = int(df.isna().sum().sum())
    if missing > 0:
        insights.append(f"Dataset contains {missing} missing values. Consider imputation.")

    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        insights.append(f"Dataset contains {duplicates} duplicate rows. Consider deduplication.")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        corr = numeric.corr().abs()
        high_corr = []
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1 :]:
                if corr.loc[col_a, col_b] >= 0.85:
                    high_corr.append((col_a, col_b))
        if high_corr:
            insights.append("High correlation detected among numeric features.")

    categorical_count = len(df.select_dtypes(exclude="number").columns)
    if categorical_count > 0:
        insights.append("Categorical columns detected. Encoding will be required for modeling.")

    if not insights:
        insights.append("No major quality risks detected. Dataset is ready for modeling.")

    return insights
