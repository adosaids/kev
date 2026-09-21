from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def choice_confidence(probabilities: Sequence[float]) -> float:
    """Return the top-label probability used by inference and selective evaluation."""
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return max(float(probability) for probability in probabilities)


@dataclass(frozen=True)
class ChoiceAnswer:
    """A closed-set decision whose selected value always comes from ``options``."""

    choice: str
    probabilities: dict[str, float]
    confidence: float

    @classmethod
    def from_logits(
        cls,
        options: Sequence[str],
        logits: Sequence[float],
        temperature: float = 1.0,
    ) -> "ChoiceAnswer":
        if not options:
            raise ValueError("options must not be empty")
        if len(options) != len(logits):
            raise ValueError("options and logits must have the same length")
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        scaled = [float(value) / temperature for value in logits]
        maximum = max(scaled)
        exponentials = [math.exp(value - maximum) for value in scaled]
        denominator = sum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        winner = max(range(len(options)), key=probabilities.__getitem__)

        return cls(
            choice=options[winner],
            probabilities=dict(zip(options, probabilities, strict=True)),
            confidence=choice_confidence(probabilities),
        )
