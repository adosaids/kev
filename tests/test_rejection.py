import pytest

from kev.rejection import fit_rejection_threshold, rejection_metrics


def test_threshold_maximizes_balanced_accuracy():
    threshold = fit_rejection_threshold(
        in_scope_scores=[0.05, 0.10, 0.20, 0.30],
        ood_scores=[0.40, 0.60, 0.80, 0.90],
    )

    assert threshold == pytest.approx(0.35)


def test_rejection_metrics_report_both_sides():
    metrics = rejection_metrics(
        in_scope_scores=[0.1, 0.7],
        ood_scores=[0.8, 0.2],
        threshold=0.5,
    )

    assert metrics == {
        "threshold": 0.5,
        "in_scope_accept_rate": 0.5,
        "false_reject_rate": 0.5,
        "ood_recall": 0.5,
        "ood_false_accept_rate": 0.5,
        "balanced_accuracy": 0.5,
    }
