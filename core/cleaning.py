"""Data cleaning logic with suggestions and smart recommendations."""

from __future__ import annotations

from typing import Literal

import pandas as pd

MissingStrategy = Literal["mean", "median", "mode", "drop"]


def suggest_cleaning_actions(df: pd.DataFrame) -> list[str]:
    """Generate cleaning suggestions based on dataset diagnostics."""
    suggestions: list[str] = []
    missing_ratio = (df.isna().sum() / max(len(df), 1)).sort_values(ascending=False)
    if df.isna().sum().sum() > 0:
        suggestions.append("Dataset contains missing values; apply imputation.")
    for col, ratio in missing_ratio.head(3).items():
        if ratio > 0.4:
            suggestions.append(f"Recommended: Drop column '{col}' due to high missingness ({ratio:.0%}).")
        elif ratio > 0.1:
            strategy = "median" if col in df.select_dtypes(include="number").columns else "mode"
            suggestions.append(f"Recommended: Use {strategy} imputation for '{col}' ({ratio:.0%} missing).")

    if df.duplicated().sum() > 0:
        suggestions.append("Dataset contains duplicate rows; remove duplicates.")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        skewed_cols = numeric.skew(numeric_only=True).abs()
        for col, skew in skewed_cols.items():
            if skew > 1.0:
                suggestions.append(f"Recommended: Use median for '{col}' due to skewness ({skew:.2f}).")
                break
        suggestions.append("Check numeric columns for outliers using IQR filtering.")
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if object_cols:
        suggestions.append("Review object columns for potential type conversion and category normalization.")
    return suggestions


def handle_missing_values(df: pd.DataFrame, strategy: MissingStrategy = "mean") -> pd.DataFrame:
    """Handle missing values using selected strategy."""
    out = df.copy()
    if strategy == "drop":
        return out.dropna()

    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue
        if strategy == "mean" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])
    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicated rows."""
    return df.drop_duplicates().reset_index(drop=True)


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> dict[str, int]:
    """Count outliers per numeric column using IQR."""
    outliers: dict[str, int] = {}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        q1, q3 = numeric[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (numeric[col] < lower) | (numeric[col] > upper)
        outliers[col] = int(mask.sum())
    return outliers


def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt automatic type correction for object columns."""
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        converted_numeric = pd.to_numeric(out[col], errors="coerce")
        if converted_numeric.notna().mean() >= 0.9:
            out[col] = converted_numeric
            continue
        converted_datetime = pd.to_datetime(out[col], errors="coerce")
        if converted_datetime.notna().mean() >= 0.9:
            out[col] = converted_datetime
    return out
