from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from kev.contracts import ChoiceAnswer
from kev.data import option_description


DEFAULT_BASE_MODEL = "microsoft/MiniLM-L12-H384-uncased"
DEFAULT_QUESTION = "What is the customer's banking intent?"


def candidate_text(question: str, option: str) -> str:
    return f"Question: {question}\nCandidate answer: {option_description(option)}"


@dataclass(frozen=True)
class CandidatePair:
    state: str
    question: str
    option: str


class CrossEncoderDecisionModel:
    """Scores caller-provided options without generating text."""

    def __init__(self, model, tokenizer, *, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        max_length: int = 128,
        trainable_head: bool = False,
    ) -> "CrossEncoderDecisionModel":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        kwargs = {"num_labels": 1} if trainable_head else {}
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        return cls(model, tokenizer, max_length=max_length)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def tokenize_pairs(self, pairs: Sequence[CandidatePair]):
        if not pairs:
            raise ValueError("pairs must not be empty")
        candidates = [
            candidate_text(pair.question, pair.option)
            for pair in pairs
        ]
        return self.tokenizer(
            [pair.state for pair in pairs],
            candidates,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @torch.inference_mode()
    def score_options(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        *,
        batch_size: int = 64,
    ) -> list[float]:
        if not options:
            raise ValueError("options must not be empty")
        self.model.eval()
        scores: list[float] = []
        for start in range(0, len(options), batch_size):
            option_batch = list(options[start : start + batch_size])
            encoded = self.tokenize_pairs(
                [CandidatePair(state, question, option) for option in option_batch]
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits.squeeze(-1)
            scores.extend(float(value) for value in logits.cpu())
        return scores

    def decide(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        *,
        temperature: float = 1.0,
        batch_size: int = 64,
    ) -> ChoiceAnswer:
        logits = self.score_options(state, question, options, batch_size=batch_size)
        return ChoiceAnswer.from_logits(options, logits, temperature=temperature)

    def save(self, output_dir: str | Path) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
