"""Recommendation and simulation engine for CDAS."""

from __future__ import annotations


def adaptive_pipeline(issues: list[dict[str, object]]) -> list[str]:
    """Build adaptive preprocessing pipeline from reasoning issues."""
    steps: list[str] = []
    issue_text = " ".join(i["issue"] for i in issues).lower()

    if "missing" in issue_text:
        steps.append("Impute missing values")
    if "duplicate" in issue_text:
        steps.append("Remove duplicate rows")
    if "correlation" in issue_text:
        steps.append("Reduce multicollinearity (drop correlated features)")
    if "class imbalance" in issue_text:
        steps.append("Use stratified training and imbalance mitigation")

    steps.append("Encode categorical features")
    steps.append("Scale numeric features")
    return list(dict.fromkeys(steps))


def model_advisor(problem_type: str, num_classes: int | None = None) -> list[dict[str, str]]:
    """Explainable model recommendations."""
    if problem_type == "classification":
        if num_classes == 2:
            return [
                {"model": "Logistic Regression", "why": "Target has 2 classes and benefits from interpretable linear boundaries."},
                {"model": "Random Forest Classifier", "why": "Handles nonlinear effects and mixed feature interactions."},
            ]
        return [
            {"model": "Random Forest Classifier", "why": "Robust for multiclass patterns and mixed feature types."},
            {"model": "Gradient Boosting Classifier", "why": "Often strong on tabular multiclass tasks."},
        ]

    return [
        {"model": "Linear Regression", "why": "Provides interpretable baseline for continuous targets."},
        {"model": "Random Forest Regressor", "why": "Captures nonlinear relationships with limited feature tuning."},
    ]


def error_simulation(issues: list[dict[str, object]]) -> list[str]:
    """Simulate consequences of skipping recommended fixes."""
    text = " ".join(i["issue"] for i in issues).lower()
    outcomes: list[str] = []

    if "missing" in text:
        outcomes.append("If missing values are ignored, model reliability can degrade and variance may increase.")
    if "correlation" in text:
        outcomes.append("If high correlation is ignored, coefficient instability and redundant features may persist.")
    if "class imbalance" in text:
        outcomes.append("If imbalance is ignored, minority classes may be systematically under-predicted.")
    if "duplicate" in text:
        outcomes.append("If duplicates are ignored, model metrics may look overly optimistic.")

    if not outcomes:
        outcomes.append("No major simulation risks detected for current dataset health.")
    return outcomes
