import os
import pandas as pd
import numpy as np
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _local_insights(df: pd.DataFrame) -> list:
    out = []
    n_missing = df.isna().sum().sum()
    out.append(f"Total missing values: {int(n_missing)}")
    num = df.select_dtypes(include=["number"])
    if not num.empty:
        out.append(f"Numeric columns: {', '.join(num.columns[:5])}")
        corr = num.corr().abs()
        high = []
        for i in corr.columns:
            for j in corr.columns:
                if i != j and corr.loc[i, j] > 0.7:
                    high.append((i, j, corr.loc[i, j]))
        if high:
            out.append("High correlations detected: " + ", ".join([f"{a}/{b}: {c:.2f}" for a, b, c in high[:5]]))
    # outliers simple
    from scipy import stats

    if not num.empty:
        zscores = np.abs(stats.zscore(num.fillna(num.mean())))
        if zscores.size:
            if zscores.ndim == 1:
                count_out = int((zscores > 3).sum())
                out.append(f"Outlier counts (per column): {num.columns[0]}:{count_out}")
            else:
                count_out = (zscores > 3).sum(axis=0)
                out.append(
                    "Outlier counts (per column): "
                    + ", ".join([f"{col}:{int(v)}" for col, v in zip(num.columns, count_out)])
                )
    return out


def generate_insights(df: pd.DataFrame) -> list:
    # Prefer OpenAI if configured
    key = None
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    prompt = None
    if key and OpenAI is not None:
        try:
            client = OpenAI(api_key=key)
            prompt = f"Provide 6 concise insights about this dataset: columns {list(df.columns[:10])}. Summarize missingness, high correlations, and suggested visualizations."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            text = resp.choices[0].message.content
            return [t.strip() for t in text.split("\n") if t.strip()]
        except Exception:
            pass
    # fallback local heuristics
    return _local_insights(df)
