from kev.data import ChoiceExample, grouped_candidates, train_validation_split


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
