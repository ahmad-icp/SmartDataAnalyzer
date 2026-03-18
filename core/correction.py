"""Intelligent categorical correction using fuzzy matching."""

from __future__ import annotations

import json
import importlib.util

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


def ai_suggest_corrections(values: list[str], api_key: str, model: str = "gpt-4.1-mini") -> dict[str, str]:
    """Use OpenAI to suggest normalized category mappings for unique values."""
    if not values:
        return {}
    if not api_key:
        raise ValueError("OpenAI API key is required.")

    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("openai package is not installed.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = (
        "Normalize the following categorical values into consistent labels. "
        "Return JSON object mapping each original value to its normalized label. "
        "Keep labels concise and consistent.\n\n"
        f"Values: {values}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=600,
    )
    content = response.output_text.strip()
    mapping = json.loads(content)
    if not isinstance(mapping, dict):
        raise ValueError("Model output is not a valid JSON object.")
    return {str(k): str(v) for k, v in mapping.items()}
