"""Scoring utilities for CDAS."""

from __future__ import annotations


def dataset_complexity_index(
    missing_ratio: float,
    n_features: int,
    correlation_max: float,
    imbalance: float | None,
) -> tuple[float, str]:
    """Compute Dataset Complexity Index on a 0-10 scale."""
    missing_component = min(3.0, missing_ratio * 10)
    feature_component = min(2.5, n_features / 20)
    correlation_component = min(2.5, correlation_max * 2.5)
    imbalance_component = 0.0 if imbalance is None else min(2.0, (1 - imbalance) * 2.0)
    score = round(min(10.0, missing_component + feature_component + correlation_component + imbalance_component), 2)

    if score < 3.5:
        label = "Low Complexity"
    elif score < 7.0:
        label = "Medium Complexity"
    else:
        label = "High Complexity"
    return score, label


def confidence_from_signal(signal_strength: float) -> float:
    """Map heuristic signal strength into confidence [0, 1]."""
    return max(0.0, min(1.0, round(signal_strength, 2)))
