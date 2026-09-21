import pytest

from kev.evaluate import validate_sample_counts


def test_evaluation_requires_non_empty_calibration_and_test_sets():
    validate_sample_counts(1, 1)

    with pytest.raises(ValueError, match="calibration_samples"):
        validate_sample_counts(0, 1)
    with pytest.raises(ValueError, match="test_samples"):
        validate_sample_counts(1, 0)

