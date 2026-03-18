"""Multi-step reasoning engine for issue detection."""

from __future__ import annotations

from core.cognitive.scoring import confidence_from_signal


def build_issue(
    issue: str,
    severity: str,
    reasoning: str,
    recommendation: str,
    confidence: float,
) -> dict[str, object]:
    """Create structured issue payload."""
    return {
        "issue": issue,
        "severity": severity,
        "reasoning": reasoning,
        "recommendation": recommendation,
        "confidence": confidence_from_signal(confidence),
    }


def reason_from_diagnostics(d: dict[str, object]) -> list[dict[str, object]]:
    """Generate structured multi-step reasoning for detected issues."""
    issues: list[dict[str, object]] = []

    missing_ratio = float(d["missing_ratio"])
    if missing_ratio > 0:
        severity = "High" if missing_ratio > 0.2 else "Medium" if missing_ratio > 0.05 else "Low"
        issues.append(
            build_issue(
                issue=f"Missing Values detected ({missing_ratio:.1%})",
                severity=severity,
                reasoning="Missing values reduce data completeness and can destabilize model estimates.",
                recommendation="Apply median/mode imputation or remove low-quality rows.",
                confidence=min(1.0, 0.6 + missing_ratio),
            )
        )

    duplicate_ratio = float(d["duplicate_ratio"])
    if duplicate_ratio > 0:
        issues.append(
            build_issue(
                issue=f"Duplicate Rows detected ({duplicate_ratio:.1%})",
                severity="Medium" if duplicate_ratio > 0.02 else "Low",
                reasoning="Duplicates bias metrics and inflate confidence in repeated patterns.",
                recommendation="Remove duplicates before modeling.",
                confidence=min(1.0, 0.5 + duplicate_ratio * 2),
            )
        )

    corr = float(d["correlation_max"])
    if corr > 0.85:
        issues.append(
            build_issue(
                issue=f"High Correlation detected (max={corr:.2f})",
                severity="Medium",
                reasoning="Strong multicollinearity can reduce interpretability and harm linear model stability.",
                recommendation="Drop one of highly correlated features or use regularized/tree-based models.",
                confidence=min(1.0, 0.55 + corr / 2),
            )
        )

    balance = d.get("target_balance")
    if balance is not None and float(balance) < 0.35:
        issues.append(
            build_issue(
                issue=f"Class Imbalance detected (ratio={float(balance):.2f})",
                severity="High" if float(balance) < 0.2 else "Medium",
                reasoning="Imbalance can produce misleading accuracy and poor minority-class recall.",
                recommendation="Use stratified split, class weighting, or sampling techniques.",
                confidence=min(1.0, 0.65 + (1 - float(balance)) / 3),
            )
        )

    return issues
