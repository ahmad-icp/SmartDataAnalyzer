"""Intelligent categorical correction using fuzzy matching."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz


def suggest_corrections(series: pd.Series, threshold: int = 85) -> list[dict[str, object]]:
    """Group similar categorical values and suggest canonical values."""
    values = [str(v).strip() for v in series.dropna().unique() if str(v).strip()]
    used: set[str] = set()
    groups: list[dict[str, object]] = []

    for base in values:
        if base in used:
            continue
        cluster = [base]
        used.add(base)
        for candidate in values:
            if candidate in used:
                continue
            if fuzz.ratio(base.lower(), candidate.lower()) >= threshold:
                cluster.append(candidate)
                used.add(candidate)

        if len(cluster) > 1:
            canonical = max(cluster, key=lambda x: (len(x), x.count("-")))
            groups.append({"original": sorted(cluster), "suggested": canonical.upper()})

    return groups


def apply_corrections(df: pd.DataFrame, column: str, corrections: list[dict[str, object]]) -> pd.DataFrame:
    """Apply accepted fuzzy correction groups to a column."""
    mapping: dict[str, str] = {}
    for item in corrections:
        suggested = str(item["suggested"])
        for original in item["original"]:
            mapping[str(original)] = suggested

    out = df.copy()
    out[column] = out[column].astype(str).map(lambda x: mapping.get(x, x))
    return out
