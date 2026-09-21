import torch

from kev.calibration import fit_temperature, multiclass_nll


def test_temperature_scaling_improves_overconfident_predictions():
    logits = torch.tensor(
        [
            [8.0, 0.0],
            [8.0, 0.0],
            [0.0, 8.0],
            [0.0, 8.0],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])

    before = multiclass_nll(logits, labels, temperature=1.0)
    temperature = fit_temperature(logits, labels)
    after = multiclass_nll(logits, labels, temperature=temperature)

    assert temperature > 1.0
    assert after < before

