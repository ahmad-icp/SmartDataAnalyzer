import pandas as pd

from core.correction import suggest_corrections


def test_suggest_corrections_groups_similar_values() -> None:
    s = pd.Series(["FAST NUCES", "Fast Nuces", "FAST-NUCES", "OTHER"])
    suggestions = suggest_corrections(s, threshold=80)
    assert any("FAST NUCES" in item["original"] for item in suggestions)
