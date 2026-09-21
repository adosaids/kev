from __future__ import annotations

from collections.abc import Sequence


def _validate_scores(scores: Sequence[float], name: str) -> list[float]:
    values = [float(score) for score in scores]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(score < 0 or score > 1 for score in values):
        raise ValueError(f"{name} must be probabilities between zero and one")
    return values


def rejection_metrics(
    in_scope_scores: Sequence[float],
    ood_scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    known = _validate_scores(in_scope_scores, "in_scope_scores")
    ood = _validate_scores(ood_scores, "ood_scores")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    false_reject_rate = sum(score >= threshold for score in known) / len(known)
    ood_recall = sum(score >= threshold for score in ood) / len(ood)
    return {
        "threshold": threshold,
        "in_scope_accept_rate": 1.0 - false_reject_rate,
        "false_reject_rate": false_reject_rate,
        "ood_recall": ood_recall,
        "ood_false_accept_rate": 1.0 - ood_recall,
        "balanced_accuracy": ((1.0 - false_reject_rate) + ood_recall) / 2.0,
    }


def fit_rejection_threshold(
    in_scope_scores: Sequence[float],
    ood_scores: Sequence[float],
) -> float:
    known = _validate_scores(in_scope_scores, "in_scope_scores")
    ood = _validate_scores(ood_scores, "ood_scores")
    ordered = sorted(set([0.0, *known, *ood, 1.0]))
    candidates = [(left + right) / 2.0 for left, right in zip(ordered, ordered[1:])]
    if not candidates:
        return 0.5
    return max(
        candidates,
        key=lambda threshold: (
            rejection_metrics(known, ood, threshold=threshold)["balanced_accuracy"],
            threshold,
        ),
    )
