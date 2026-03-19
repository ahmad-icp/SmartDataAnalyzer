import pandas as pd

from core.intelligence_engine import generate_intelligence_report


def test_generate_intelligence_report_detects_missing() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    insights = generate_intelligence_report(df)
    assert any("missing values" in item.lower() for item in insights)
