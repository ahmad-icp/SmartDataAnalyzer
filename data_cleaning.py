"""Data cleaning utilities for Smart Data Analyzer."""

import pandas as pd


def remove_missing_values(df: pd.DataFrame, how: str = "any") -> pd.DataFrame:
    """Remove rows with missing values.

    Args:
        df: Input DataFrame.
        how: 'any' or 'all' (passed through to dropna).

    Returns:
        DataFrame with missing rows removed.
    """

    return df.dropna(how=how).copy()


def fill_missing_values(df: pd.DataFrame, method: str = "Mean") -> pd.DataFrame:
    """Fill missing values using mean, median, or mode.

    Args:
        df: Input DataFrame.
        method: One of 'Mean', 'Median', 'Mode'.

    Returns:
        DataFrame with missing values filled.
    """

    method = method.lower()
    out = df.copy()

    if method == "mean":
        return out.fillna(out.mean(numeric_only=True))
    if method == "median":
        return out.fillna(out.median(numeric_only=True))
    if method == "mode":
        modes = out.mode(dropna=True)
        if not modes.empty:
            return out.fillna(modes.iloc[0])
        return out

    raise ValueError(f"Unknown fill method: {method}")


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame."""

    return df.drop_duplicates(subset=subset).copy()


def drop_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Drop selected columns from the DataFrame."""

    if not columns:
        return df
    return df.drop(columns=columns, errors="ignore").copy()


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns according to a mapping."""

    return df.rename(columns=rename_map).copy()


def convert_column_type(df: pd.DataFrame, column: str, dtype: str) -> pd.DataFrame:
    """Convert a column to a specified dtype."""

    out = df.copy()
    if column not in out.columns:
        return out

    if dtype == "numeric":
        out[column] = pd.to_numeric(out[column], errors="coerce")
    elif dtype == "string":
        out[column] = out[column].astype(str)
    elif dtype == "category":
        out[column] = out[column].astype("category")
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return out


def filter_rows(df: pd.DataFrame, column: str, operator: str, value: str) -> pd.DataFrame:
    """Filter rows using a simple condition.

    Supports common operators and a "contains" operation for strings.
    """

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    series = df[column]
    if operator == "contains":
        return df[series.astype(str).str.contains(str(value), na=False)].copy()

    try:
        test_value = float(value)
    except ValueError:
        test_value = value

    if operator == "==":
        return df[series == test_value].copy()
    if operator == "!=":
        return df[series != test_value].copy()
    if operator == ">":
        return df[series > test_value].copy()
    if operator == "<":
        return df[series < test_value].copy()
    if operator == ">=":
        return df[series >= test_value].copy()
    if operator == "<=":
        return df[series <= test_value].copy()

    raise ValueError(f"Unsupported operator: {operator}")
