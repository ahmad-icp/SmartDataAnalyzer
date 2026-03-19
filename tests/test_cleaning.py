import pandas as pd

from core.cleaning import handle_missing_values, remove_duplicates


def test_handle_missing_mean() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = handle_missing_values(df, strategy="mean")
    assert out["a"].isna().sum() == 0


def test_remove_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2]})
    out = remove_duplicates(df)
    assert len(out) == 2
