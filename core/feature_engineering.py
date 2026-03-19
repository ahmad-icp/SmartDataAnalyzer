"""Feature engineering pipeline components and recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureEngineeringResult:
    """Feature engineering outputs and recommendations."""

    transformed: pd.DataFrame
    dropped_low_variance: list[str]
    high_correlation_pairs: list[tuple[str, str, float]]
    recommendations: list[str]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return df.copy()
    return pd.get_dummies(df, columns=list(cat_cols), drop_first=False)


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard scale numeric columns."""
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        out[num_cols] = scaler.fit_transform(out[num_cols])
    return out


def remove_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove low variance numeric features."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return df.copy(), []

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric)
    kept = numeric.columns[selector.get_support()].tolist()
    dropped = [c for c in numeric.columns if c not in kept]
    out = pd.concat([df.drop(columns=numeric.columns), numeric[kept]], axis=1)
    return out, dropped


def suggest_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
) -> list[tuple[str, str, float]]:
    """Find highly correlated feature pairs."""
    corr = df.select_dtypes(include="number").corr().abs()
    pairs: list[tuple[str, str, float]] = []
    if corr.empty:
        return pairs
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if value >= threshold:
                pairs.append((col_a, col_b, float(value)))
    return pairs


def build_feature_recommendations(
    dropped_low_variance: list[str],
    high_correlation_pairs: list[tuple[str, str, float]],
) -> list[str]:
    """Generate recommendation text based on feature diagnostics."""
    recommendations: list[str] = []
    if dropped_low_variance:
        recommendations.append(
            f"Recommended: Keep low-variance features dropped ({', '.join(dropped_low_variance[:5])})."
        )
    for col_a, col_b, corr in high_correlation_pairs[:3]:
        recommendations.append(
            f"Recommended: Consider dropping either '{col_a}' or '{col_b}' (correlation={corr:.2f})."
        )
    if high_correlation_pairs:
        recommendations.append("Recommended: Tree-based models are suitable for correlated feature spaces.")
    return recommendations


def run_feature_engineering(df: pd.DataFrame) -> FeatureEngineeringResult:
    """Run basic feature engineering pipeline."""
    encoded = encode_features(df)
    no_low_var, dropped = remove_low_variance_features(encoded)
    scaled = scale_numeric_features(no_low_var)
    corr_pairs = suggest_highly_correlated_features(scaled)
    recommendations = build_feature_recommendations(dropped, corr_pairs)
    return FeatureEngineeringResult(scaled, dropped, corr_pairs, recommendations)
