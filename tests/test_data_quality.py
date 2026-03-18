import pandas as pd

from core.data_quality import compute_data_quality_score


def test_data_quality_score_returns_expected_payload() -> None:
    df = pd.DataFrame(
        {
            "num": [1, 2, 2, 100],
            "cat": ["A", "a", "A", "B"],
        }
    )
    report = compute_data_quality_score(df)
    assert "score" in report
    assert "penalties" in report
    assert 0 <= report["score"] <= 100
