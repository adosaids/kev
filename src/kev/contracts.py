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
    rejected: bool = False
    rejection_score: float | None = None

    @classmethod
    def from_logits(
        cls,
        options: Sequence[str],
        logits: Sequence[float],
        temperature: float = 1.0,
        abstain_option: str | None = None,
        abstain_threshold: float | None = None,
    ) -> "ChoiceAnswer":
        if not options:
            raise ValueError("options must not be empty")
        if len(options) != len(logits):
            raise ValueError("options and logits must have the same length")
        if len(set(options)) != len(options):
            raise ValueError("options must be unique")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if (abstain_option is None) != (abstain_threshold is None):
            raise ValueError("abstain_option and abstain_threshold must be supplied together")
        if abstain_option is not None and abstain_option not in options:
            raise ValueError("abstain_option must be one of the options")
        if abstain_threshold is not None and not 0 <= abstain_threshold <= 1:
            raise ValueError("abstain_threshold must be between zero and one")

        scaled = [float(value) / temperature for value in logits]
        maximum = max(scaled)
        exponentials = [math.exp(value - maximum) for value in scaled]
        denominator = sum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        rejected = False
        rejection_score = None
        if abstain_option is not None:
            abstain_index = options.index(abstain_option)
            best_known_logit = max(
                value for index, value in enumerate(scaled) if index != abstain_index
            )
            margin = scaled[abstain_index] - best_known_logit
            if margin >= 0:
                rejection_score = 1.0 / (1.0 + math.exp(-margin))
            else:
                exponential = math.exp(margin)
                rejection_score = exponential / (1.0 + exponential)
            rejected = rejection_score >= abstain_threshold
            if rejected:
                winner = abstain_index
            else:
                winner = max(
                    (index for index in range(len(options)) if index != abstain_index),
                    key=probabilities.__getitem__,
                )
        else:
            winner = max(range(len(options)), key=probabilities.__getitem__)

        return cls(
            choice=options[winner],
            probabilities=dict(zip(options, probabilities, strict=True)),
            confidence=probabilities[winner],
            rejected=rejected,
            rejection_score=rejection_score,
        )
