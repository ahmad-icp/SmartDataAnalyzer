"""Intelligent categorical correction using local fuzzy and heuristic logic."""

from __future__ import annotations

from collections import Counter

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
    """Apply accepted correction groups to a column."""
    mapping: dict[str, str] = {}
    for item in corrections:
        suggested = str(item["suggested"])
        for original in item["original"]:
            mapping[str(original)] = suggested

    out = df.copy()
    out[column] = out[column].astype(str).map(lambda x: mapping.get(x, x))
    return out


def ai_suggest_corrections(values: list[str], api_key: str = "", model: str = "") -> dict[str, str]:
    """Local heuristic normalization mapping (API-free compatibility wrapper)."""
    del api_key, model
    clean_values = [str(v).strip() for v in values if str(v).strip()]
    if not clean_values:
        return {}

    normalized = [v.upper().replace("_", " ").replace("-", " ") for v in clean_values]
    mapping: dict[str, str] = {}
    counter = Counter(normalized)

    for original, norm in zip(clean_values, normalized):
        canonical = max((n for n in counter if fuzz.ratio(norm, n) >= 85), key=lambda n: (counter[n], len(n)))
        mapping[original] = canonical

    return mapping
