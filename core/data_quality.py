"""Data quality scoring module."""

from __future__ import annotations

import pandas as pd

from core.cleaning import detect_outliers_iqr
from core.correction import suggest_corrections


MAX_PENALTY_MISSING = 35
MAX_PENALTY_DUPLICATES = 20
MAX_PENALTY_OUTLIERS = 25
MAX_PENALTY_INCONSISTENCY = 20


def compute_data_quality_score(df: pd.DataFrame) -> dict[str, object]:
    """Compute a data quality score (0-100) with breakdown penalties."""
    n_rows = max(len(df), 1)
    n_cells = max(df.shape[0] * df.shape[1], 1)

    missing_ratio = float(df.isna().sum().sum() / n_cells)
    duplicate_ratio = float(df.duplicated().sum() / n_rows)

    outlier_counts = detect_outliers_iqr(df)
    total_outliers = sum(outlier_counts.values())
    numeric_cells = max(df.select_dtypes(include="number").size, 1)
    outlier_ratio = float(total_outliers / numeric_cells)

    categorical_cols = df.select_dtypes(include="object").columns
    inconsistency_count = 0
    possible_groups = 0
    for col in categorical_cols:
        suggestions = suggest_corrections(df[col])
        inconsistency_count += len(suggestions)
        possible_groups += max(df[col].nunique(dropna=True), 1)

    inconsistency_ratio = inconsistency_count / max(possible_groups, 1)

    penalties = {
        "missing_values": round(min(MAX_PENALTY_MISSING, missing_ratio * 100), 2),
        "duplicates": round(min(MAX_PENALTY_DUPLICATES, duplicate_ratio * 100), 2),
        "outliers": round(min(MAX_PENALTY_OUTLIERS, outlier_ratio * 100), 2),
        "inconsistent_categories": round(
            min(MAX_PENALTY_INCONSISTENCY, inconsistency_ratio * 100), 2
        ),
    }

    total_penalty = sum(penalties.values())
    score = max(0.0, min(100.0, 100.0 - total_penalty))

    return {
        "score": round(score, 2),
        "label": f"Data Quality Score: {round(score):.0f}/100",
        "penalties": penalties,
    }
