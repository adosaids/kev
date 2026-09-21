import pytest

from kev.contracts import ChoiceAnswer


def test_choice_answer_can_only_select_a_declared_option():
    answer = ChoiceAnswer.from_logits(
        options=["browser", "shell", "ask_user"],
        logits=[0.1, 2.0, -1.0],
    )

    assert answer.choice == "shell"
    assert set(answer.probabilities) == {"browser", "shell", "ask_user"}
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_choice_answer_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same length"):
        ChoiceAnswer.from_logits(options=["browser"], logits=[0.1, 0.2])


def test_choice_answer_applies_calibrated_rejection_threshold():
    rejected = ChoiceAnswer.from_logits(
        options=["browser", "shell", "none_of_above"],
        logits=[0.0, 1.0, 2.0],
        abstain_option="none_of_above",
        abstain_threshold=0.7,
    )
    accepted = ChoiceAnswer.from_logits(
        options=["browser", "shell", "none_of_above"],
        logits=[0.0, 1.0, 2.0],
        abstain_option="none_of_above",
        abstain_threshold=0.8,
    )

    assert rejected.choice == "none_of_above"
    assert rejected.rejected is True
    assert rejected.rejection_score == pytest.approx(0.731058, rel=1e-5)
    assert accepted.choice == "shell"
    assert accepted.rejected is False


def test_rejection_score_is_invariant_to_irrelevant_extra_options():
    compact = ChoiceAnswer.from_logits(
        ["shell", "none_of_above"],
        [1.0, 2.0],
        abstain_option="none_of_above",
        abstain_threshold=0.5,
    )
    expanded = ChoiceAnswer.from_logits(
        ["shell", "irrelevant", "none_of_above"],
        [1.0, -100.0, 2.0],
        abstain_option="none_of_above",
        abstain_threshold=0.5,
    )

    assert compact.rejection_score == pytest.approx(expanded.rejection_score)
