import pandas as pd
import pytest

from core.model_selection import run_model_selection


def test_model_selection_returns_table() -> None:
    df = pd.DataFrame(
        {
            "x1": [1, 2, 3, 4, 5, 6, 7, 8],
            "x2": [2, 1, 2, 1, 2, 1, 2, 1],
            "target": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    result = run_model_selection(df, "target")
    assert not result.performance_table.empty
    assert isinstance(result.best_model_name, str)


def test_model_selection_rejects_too_few_rows() -> None:
    df = pd.DataFrame({"x1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match="At least 5 rows"):
        run_model_selection(df, "target")
