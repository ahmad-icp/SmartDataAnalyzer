from core.eda import dataset_overview


def test_overview_shape() -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2]})
    overview = dataset_overview(df)
    assert overview["shape"] == (2, 1)
