import pandas as pd
from modules.cleaning_tools import fill_missing, standardize_text_columns


def test_fill_missing_mean():
    df = pd.DataFrame({"a": [1, 2, None, 4]})
    out = fill_missing(df, strategy="mean")
    assert out["a"].isna().sum() == 0
    assert abs(out["a"].iloc[2] - (7 / 3)) < 1e-6


def test_standardize_text():
    df = pd.DataFrame({"c": ["A ", "b", "C!"]})
    out = standardize_text_columns(df, ["c"])
    assert out["c"].tolist() == ["a", "b", "c"]
