from __future__ import annotations

import torch
import torch.nn.functional as functional


def multiclass_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return float(functional.cross_entropy(logits / temperature, labels).item())


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    steps: int = 400,
) -> float:
    """Fit one scalar temperature on held-out logits using a stable grid search."""
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [examples, classes] and labels must be [examples]")
    if steps < 2 or not 0 < minimum < maximum:
        raise ValueError("invalid temperature search range")

    candidates = torch.logspace(
        torch.log10(torch.tensor(minimum)),
        torch.log10(torch.tensor(maximum)),
        steps,
        device=logits.device,
    )
    losses = torch.stack(
        [functional.cross_entropy(logits / temperature, labels) for temperature in candidates]
    )
    return float(candidates[losses.argmin()].item())

