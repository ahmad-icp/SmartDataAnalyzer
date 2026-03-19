import pandas as pd

from core.cognitive.analyzer import analyze_dataset
from core.cognitive.reasoning import reason_from_diagnostics
from core.cognitive.scoring import dataset_complexity_index


def test_analyze_dataset_and_reasoning() -> None:
    df = pd.DataFrame({"a": [1, None, 3, 3], "b": [1, 2, 3, 3], "target": [0, 0, 1, 1]})
    diag = analyze_dataset(df, target_column="target")
    issues = reason_from_diagnostics(diag)
    assert diag["n_rows"] == 4
    assert len(issues) >= 1


def test_dataset_complexity_index_bounds() -> None:
    score, label = dataset_complexity_index(0.2, 20, 0.9, 0.3)
    assert 0 <= score <= 10
    assert label in {"Low Complexity", "Medium Complexity", "High Complexity"}
