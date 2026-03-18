"""Automated human-readable insight engine."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_data_insights(
    df: pd.DataFrame,
    corr_threshold: float = 0.85,
    normalized_columns: list[str] | None = None,
    max_correlation_insights: int = 8,
) -> list[str]:
    """Generate data-centric insights from quality and distribution checks."""
    insights: list[str] = []
    normalized_columns = normalized_columns or []

    missing_ratio = (df.isna().sum() / max(len(df), 1)).sort_values(ascending=False)
    for col, ratio in missing_ratio.items():
        if ratio >= 0.2:
            insights.append(
                f"Column '{col}' has {ratio:.0%} missing values. Consider imputation or targeted cleanup."
            )

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        skew = numeric.skew(numeric_only=True)
        for col, value in skew.items():
            if abs(value) > 1:
                insights.append(f"Column '{col}' is highly skewed ({value:.2f}); robust transformations may help.")

        corr = numeric.corr(numeric_only=True)
        corr_insights_count = 0
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1 :]:
                value = corr.loc[col_a, col_b]
                if abs(value) >= corr_threshold:
                    insights.append(
                        f"Features '{col_a}' and '{col_b}' are highly correlated ({value:.2f}); consider pruning one."
                    )
                    corr_insights_count += 1
                    if corr_insights_count >= max_correlation_insights:
                        insights.append(
                            "Additional high-correlation pairs were detected; review the full correlation matrix."
                        )
                        break
            if corr_insights_count >= max_correlation_insights:
                break

    for col in normalized_columns:
        insights.append(f"Column '{col}' had inconsistent categorical values and was normalized.")

    if not insights:
        insights.append("Dataset quality appears stable with no major red flags detected.")
    return insights


def generate_model_insights(model_result: Any) -> list[str]:
    """Generate model-centric insights from performance signals."""
    insights: list[str] = []
    best_model = model_result.best_model_name
    metric_name = model_result.metric_name
    test_score = model_result.best_test_score
    train_score = model_result.best_train_score

    if "Random Forest" in best_model or "Gradient Boosting" in best_model:
        insights.append(f"{best_model} performed best, suggesting non-linear patterns in the data.")
    else:
        insights.append(f"{best_model} performed best, indicating simpler linear relationships are effective.")

    if metric_name == "accuracy" and test_score < 0.65:
        insights.append("Model accuracy is relatively low; dataset may be noisy, imbalanced, or under-informative.")
    if metric_name == "rmse" and test_score > model_result.y_test.std():
        insights.append("RMSE is high versus target spread; additional feature engineering may be required.")

    if metric_name == "accuracy" and (train_score - test_score) > 0.12:
        insights.append("Potential overfitting detected: train accuracy is notably higher than test accuracy.")
    if metric_name == "rmse" and (test_score - train_score) > 0.2 * max(train_score, 1e-8):
        insights.append("Potential overfitting detected: test RMSE is substantially worse than train RMSE.")

    return insights


def generate_feature_insights(feature_importance: pd.DataFrame | None = None) -> list[str]:
    """Generate feature-level insights from feature importance outputs."""
    if feature_importance is None or feature_importance.empty:
        return ["Feature importance not available yet. Run explainability to unlock feature insights."]

    top_feature = str(feature_importance.iloc[0]["feature"])
    top_value = float(feature_importance.iloc[0]["importance"])
    return [f"Feature '{top_feature}' is currently the most influential predictor (importance={top_value:.4f})."]


def generate_all_insights(
    df: pd.DataFrame,
    model_result: Any | None = None,
    feature_importance: pd.DataFrame | None = None,
    normalized_columns: list[str] | None = None,
) -> list[str]:
    """Combine data, model, and feature insights into a single list for UI rendering."""
    insights = generate_data_insights(df, normalized_columns=normalized_columns)
    if model_result is not None:
        insights.extend(generate_model_insights(model_result))
    insights.extend(generate_feature_insights(feature_importance))
    return insights
