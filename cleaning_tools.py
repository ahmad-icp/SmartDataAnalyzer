import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
from collections import defaultdict


def fill_missing(df: pd.DataFrame, strategy: str = "none", custom=None) -> pd.DataFrame:
    df = df.copy()
    if strategy in ("none", "None"):
        return df
    if strategy == "mean":
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].fillna(df[c].mean())
    elif strategy == "median":
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].fillna(df[c].median())
    elif strategy == "mode":
        for c in df.columns:
            mode = df[c].mode()
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])
    elif strategy == "custom":
        df = df.fillna(custom)
    return df


def remove_duplicates(df: pd.DataFrame, fuzzy: bool = False) -> pd.DataFrame:
    if not fuzzy:
        return df.drop_duplicates().reset_index(drop=True)
    # fuzzy removal: naive approach by stringifying rows
    keys = df.astype(str).agg("||".join, axis=1)
    keep = []
    seen = []
    for i, k in enumerate(keys):
        found = False
        for s in seen:
            if fuzz.ratio(k, s) > 95:
                found = True
                break
        if not found:
            seen.append(k)
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def standardize_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.lower()
                .apply(lambda s: re.sub(r"[^\w\s]", "", s))
            )
    return df


def suggest_fuzzy_matches(df: pd.DataFrame, threshold: int = 85) -> dict:
    """Return suggestions for near-duplicate category values per object column.

    Output: {col: [(canonical, variant, score), ...]}
    """
    suggestions = {}
    for c in df.select_dtypes(include=[object]).columns:
        vals = pd.Series(df[c].dropna().unique()).astype(str)
        pairs = []
        for i, a in enumerate(vals):
            for j in range(i + 1, len(vals)):
                b = vals.iloc[j]
                score = fuzz.token_sort_ratio(a, b)
                if score >= threshold:
                    pairs.append((a, b, int(score)))
        if pairs:
            suggestions[c] = pairs
    return suggestions


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """mapping: {col: {old_value: new_value}}"""
    df = df.copy()
    for c, mp in mapping.items():
        if c in df.columns:
            df[c] = df[c].replace(mp)
    return df


def convert_types(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    for col, to in mapping.items():
        if col not in df.columns:
            continue
        if to == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif to == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif to == "category":
            df[col] = df[col].astype("category")
        elif to == "text":
            df[col] = df[col].astype(str)
    return df
