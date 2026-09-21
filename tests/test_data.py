from kev.data import (
    NONE_OF_ABOVE,
    ChoiceExample,
    abstaining_grouped_candidates,
    grouped_candidates,
    train_validation_split,
)


def test_grouped_candidates_have_one_positive_and_unique_negatives():
    example = ChoiceExample(text="Where is my card?", label="card_arrival")
    labels = ["card_arrival", "cash_withdrawal", "exchange_rate", "beneficiary_not_allowed"]

    group = grouped_candidates(example, labels, negatives=2, seed=7)

    assert len(group.options) == 3
    assert len(set(group.options)) == 3
    assert group.options[group.target] == "card_arrival"


def test_train_validation_split_is_disjoint_and_stratified():
    examples = [
        ChoiceExample(text=f"a-{index}", label="a") for index in range(10)
    ] + [ChoiceExample(text=f"b-{index}", label="b") for index in range(10)]

    train, validation = train_validation_split(examples, fraction=0.2, seed=7)

    assert len(train) == 16
    assert len(validation) == 4
    assert {example.label for example in validation} == {"a", "b"}
    assert {example.text for example in train}.isdisjoint(
        {example.text for example in validation}
    )


def test_abstaining_groups_train_none_as_negative_for_known_examples():
    example = ChoiceExample("Where is my card?", "card_arrival")
    group = abstaining_grouped_candidates(
        example, ["card_arrival", "cash_withdrawal", "exchange_rate"], negatives=2, seed=7
    )

    assert len(group.options) == 3
    assert NONE_OF_ABOVE in group.options
    assert group.options[group.target] == "card_arrival"


def test_abstaining_groups_train_none_as_positive_for_ood_examples():
    example = ChoiceExample("Who won the football game?", NONE_OF_ABOVE)
    group = abstaining_grouped_candidates(
        example, ["card_arrival", "cash_withdrawal", "exchange_rate"], negatives=2, seed=7
    )

    assert len(group.options) == 3
    assert group.options[group.target] == NONE_OF_ABOVE
