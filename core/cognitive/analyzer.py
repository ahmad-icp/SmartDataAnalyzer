"""Statistical diagnostics for Cognitive Data Analysis System."""

from __future__ import annotations

import pandas as pd


def analyze_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict[str, object]:
    """Return core diagnostics used by reasoning and recommendations."""
    n_rows, n_cols = df.shape
    missing_ratio = float(df.isna().sum().sum() / max(n_rows * n_cols, 1))
    duplicate_ratio = float(df.duplicated().sum() / max(n_rows, 1))

    numeric = df.select_dtypes(include="number")
    correlation_max = 0.0
    if not numeric.empty and numeric.shape[1] > 1:
        corr = numeric.corr().abs()
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1 :]:
                correlation_max = max(correlation_max, float(corr.loc[col_a, col_b]))

    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    numeric_cols = numeric.columns.tolist()

    target_balance = None
    if target_column and target_column in df.columns:
        counts = df[target_column].value_counts(dropna=False)
        if not counts.empty:
            target_balance = float(counts.min() / max(counts.max(), 1))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
        "correlation_max": correlation_max,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "target_balance": target_balance,
    }
