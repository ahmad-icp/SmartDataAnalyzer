"""Exploratory data analysis module."""

from __future__ import annotations

from typing import Any

import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return shape and column dtypes."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value count and ratio by column."""
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "missing_count": missing,
            "missing_ratio": (missing / len(df)).round(4),
        }
    ).sort_values("missing_count", ascending=False)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric feature correlation matrix."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True)


def numeric_distributions(df: pd.DataFrame, bins: int = 20) -> dict[str, pd.Series]:
    """Return histogram counts for each numeric column as structured output."""
    distributions: dict[str, pd.Series] = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        distributions[col] = pd.cut(numeric[col].dropna(), bins=bins).value_counts().sort_index()
    return distributions
