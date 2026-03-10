"""Statistics utilities for Smart Data Analyzer."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""

    return df.describe().transpose()


def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""

    return df.corr()


def compute_covariance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute covariance matrix for numeric columns."""

    return df.cov()


def compute_regression(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    """Fit a simple linear regression model and return metrics.

    Args:
        df: DataFrame containing the data.
        x_col: Predictor column name.
        y_col: Target column name.

    Returns:
        A dict containing the summary, coefficients, r2_score, and mse.
    """

    X = df[[x_col]].dropna()
    y = df[y_col].loc[X.index]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    summary = (
        f"Linear regression predicting {y_col} from {x_col}\n"
        f"Intercept: {model.intercept_:.4f}\n"
        f"Coefficient: {model.coef_[0]:.4f}\n"
    )

    return {
        "summary": summary,
        "coefficients": {"intercept": model.intercept_, "slope": model.coef_[0]},
        "r2_score": r2,
        "mse": mse,
    }


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers in a column using Z-score."""

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    values = df[column]
    mean = values.mean()
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std) or np.isclose(std, 0):
        return df.iloc[0:0]

    zscores = (values - mean) / std
    mask = zscores.abs() > threshold
    return df[mask].copy()


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Detect outliers using the IQR method."""

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    values = df[column].dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    mask = (df[column] < lower) | (df[column] > upper)
    return df[mask].copy()
