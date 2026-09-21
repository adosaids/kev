import pytest

from kev.metrics import classification_metrics, coverage_at_error


def test_classification_metrics_use_the_full_probability_distribution():
    probabilities = [[0.8, 0.2], [0.4, 0.6]]
    labels = [0, 1]

    metrics = classification_metrics(probabilities, labels, bins=2)

    assert metrics["accuracy"] == 1.0
    assert metrics["brier"] == pytest.approx(0.2)
    assert metrics["nll"] == pytest.approx(0.3669845875)
    assert metrics["ece"] == pytest.approx(0.3)


def test_coverage_at_error_keeps_the_largest_safe_prefix():
    probabilities = [
        [0.99, 0.01],
        [0.05, 0.95],
        [0.20, 0.80],
        [0.70, 0.30],
    ]
    labels = [0, 1, 0, 0]

    result = coverage_at_error(probabilities, labels, max_error=0.05)

    assert result["coverage"] == pytest.approx(0.5)
    assert result["accepted"] == 2
    assert result["errors"] == 0

