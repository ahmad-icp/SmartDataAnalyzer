"""Helper utilities for app-wide operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def infer_task_type(target: pd.Series) -> str:
    """Infer whether the task is classification or regression."""
    if target.dtype == "object" or target.nunique(dropna=True) < 20:
        return "classification"
    return "regression"


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean copy with normalized column names."""
    clean = df.copy()
    clean.columns = [str(c).strip().replace(" ", "_") for c in clean.columns]
    return clean


def to_serializable_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy scalars to native Python values."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out
