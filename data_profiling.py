from typing import Optional
import pandas as pd


def run_profile(df: pd.DataFrame, minimal: bool = True) -> str:
    """Generate an HTML profiling report from a DataFrame."""
    try:
        from ydata_profiling import ProfileReport
    except Exception as e:
        raise ImportError("ydata-profiling is required to generate profiling reports.") from e

    profile = ProfileReport(df, minimal=minimal)
    return profile.to_html()


__all__ = ["run_profile"]
