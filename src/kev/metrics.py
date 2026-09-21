from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be non-empty and have the same length")
    width = len(probabilities[0])
    if width < 2:
        raise ValueError("at least two classes are required")
    for row, label in zip(probabilities, labels, strict=True):
        if len(row) != width:
            raise ValueError("all probability rows must have the same length")
        if not 0 <= label < width:
            raise ValueError("label is outside the probability vector")
        if any(value < 0 or value > 1 for value in row):
            raise ValueError("probabilities must be between zero and one")
        if not math.isclose(sum(row), 1.0, abs_tol=1e-5):
            raise ValueError("each probability row must sum to one")


def classification_metrics(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    bins: int = 10,
) -> dict[str, float]:
    _validate(probabilities, labels)
    if bins < 1:
        raise ValueError("bins must be positive")

    predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    confidences = [max(row) for row in probabilities]
    correct = [int(prediction == label) for prediction, label in zip(predictions, labels, strict=True)]
    count = len(labels)

    brier = sum(
        sum((probability - int(index == label)) ** 2 for index, probability in enumerate(row))
        for row, label in zip(probabilities, labels, strict=True)
    ) / count
    nll = -sum(
        math.log(max(row[label], 1e-12))
        for row, label in zip(probabilities, labels, strict=True)
    ) / count

    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            members = [
                index
                for index, confidence in enumerate(confidences)
                if lower <= confidence <= upper
            ]
        else:
            members = [
                index
                for index, confidence in enumerate(confidences)
                if lower <= confidence < upper
            ]
        if members:
            bin_accuracy = sum(correct[index] for index in members) / len(members)
            bin_confidence = sum(confidences[index] for index in members) / len(members)
            ece += len(members) / count * abs(bin_accuracy - bin_confidence)

    return {
        "accuracy": sum(correct) / count,
        "brier": brier,
        "nll": nll,
        "ece": ece,
    }


def coverage_at_error(
    probabilities: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    max_error: float = 0.05,
) -> dict[str, float | int | None]:
    _validate(probabilities, labels)
    if not 0 <= max_error <= 1:
        raise ValueError("max_error must be between zero and one")

    ranked = sorted(
        (
            (max(row), int(max(range(len(row)), key=row.__getitem__) != label))
            for row, label in zip(probabilities, labels, strict=True)
        ),
        reverse=True,
    )

    best_accepted = 0
    best_errors = 0
    errors = 0
    for accepted, (_, is_error) in enumerate(ranked, start=1):
        errors += is_error
        if errors / accepted <= max_error:
            best_accepted = accepted
            best_errors = errors

    threshold = ranked[best_accepted - 1][0] if best_accepted else None
    return {
        "coverage": best_accepted / len(ranked),
        "accepted": best_accepted,
        "errors": best_errors,
        "threshold": threshold,
    }
