import pandas as pd
import numpy as np


def generate_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # date parts
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[f"{c}_year"] = df[c].dt.year
            df[f"{c}_month"] = df[c].dt.month
            df[f"{c}_day"] = df[c].dt.day
            df[f"{c}_weekday"] = df[c].dt.weekday
    # log transform for positives
    for c in df.select_dtypes(include=["number"]).columns:
        if (df[c] > 0).all():
            df[f"{c}_log"] = np.log1p(df[c])
    # normalize numeric
    for c in df.select_dtypes(include=["number"]).columns:
        mean = df[c].mean()
        std = df[c].std()
        if std and not np.isnan(std):
            df[f"{c}_z"] = (df[c] - mean) / std
    # simple group aggregates: top categorical cols
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if cat_cols and num_cols:
        c = cat_cols[0]
        g = df.groupby(c)[num_cols].transform("mean")
        for col in num_cols:
            df[f"{c}_{col}_grpmean"] = g[col]
    return df
