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

