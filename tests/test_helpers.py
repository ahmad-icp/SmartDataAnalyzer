import pandas as pd

from utils.helpers import infer_task_type


def test_infer_task_type_classification_for_low_cardinality_numeric() -> None:
    target = pd.Series([0, 1, 0, 1, 2, 1, 0])
    assert infer_task_type(target) == "classification"


def test_infer_task_type_regression_for_high_cardinality_numeric() -> None:
    target = pd.Series(list(range(30)))
    assert infer_task_type(target) == "regression"
