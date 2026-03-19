import pandas as pd

from core.insights import generate_data_insights


def test_generate_data_insights_flags_missing_and_correlation() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, None, 4, None],
            "b": [10, 20, 30, 40, 50],
            "c": [10, 20, 30, 40, 50],
        }
    )
    insights = generate_data_insights(df)
    assert any("missing values" in item for item in insights)
    assert any("highly correlated" in item for item in insights)


def test_generate_data_insights_caps_correlation_messages() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [3, 6, 9, 12, 15],
            "d": [4, 8, 12, 16, 20],
            "e": [5, 10, 15, 20, 25],
        }
    )
    insights = generate_data_insights(df, max_correlation_insights=2)
    assert any("Additional high-correlation pairs" in item for item in insights)
