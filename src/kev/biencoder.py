from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from kev.contracts import ChoiceAnswer
from kev.model import DEFAULT_BASE_MODEL, candidate_text


BIENCODER_CONFIG = "kev_biencoder_config.json"
BIENCODER_HEAD = "decision_head.safetensors"


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def l2_normalize(vectors: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vectors, p=2, dim=-1)


@dataclass(frozen=True)
class CachedOptions:
    owner: object
    question: str
    options: tuple[str, ...]
    vectors: torch.Tensor

    def validate(self, owner: object, question: str, options: Sequence[str]) -> None:
        if self.owner is not owner:
            raise ValueError("cached options were encoded by a different model instance")
        if self.question != question:
            raise ValueError("cached options were encoded for a different question")
        if self.options != tuple(options):
            raise ValueError("cached options do not match the ordered option list")


class BiEncoderCore(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        hidden_size = int(encoder.config.hidden_size)
        self.projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**encoded).last_hidden_state
        pooled = masked_mean(output, encoded["attention_mask"])
        return l2_normalize(self.projection(pooled))

    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


class BiEncoderDecisionModel:
    """A shared encoder with reusable option vectors and closed-set Choice output."""

    def __init__(self, model: BiEncoderCore, tokenizer, *, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._cache_owner = object()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path = DEFAULT_BASE_MODEL,
        *,
        projection_dim: int = 256,
        max_length: int = 128,
    ) -> "BiEncoderDecisionModel":
        path = Path(model_name_or_path)
        config_path = path / BIENCODER_CONFIG
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tokenizer = AutoTokenizer.from_pretrained(path)
            encoder = AutoModel.from_pretrained(path / "encoder")
            model = BiEncoderCore(encoder, projection_dim=int(config["projection_dim"]))
            state = load_file(path / BIENCODER_HEAD)
            model.projection.load_state_dict({"weight": state["projection.weight"]})
            model.logit_scale.data.copy_(state["logit_scale"])
            return cls(model, tokenizer, max_length=max_length)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        encoder = AutoModel.from_pretrained(model_name_or_path)
        return cls(BiEncoderCore(encoder, projection_dim), tokenizer, max_length=max_length)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def tokenize_texts(self, texts: Sequence[str]):
        if not texts:
            raise ValueError("texts must not be empty")
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def encode_tokenized(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        return self.model.encode(encoded)

    def _encode_texts(self, texts: Sequence[str], *, batch_size: int) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must not be empty")
        self.model.eval()
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            chunks.append(self.encode_tokenized(self.tokenize_texts(texts[start : start + batch_size])))
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def encode_options(
        self,
        question: str,
        options: Sequence[str],
        *,
        batch_size: int = 128,
    ) -> CachedOptions:
        vectors = self._encode_texts(
            [candidate_text(question, option) for option in options],
            batch_size=batch_size,
        )
        return CachedOptions(self._cache_owner, question, tuple(options), vectors)

    @torch.inference_mode()
    def encode_states(self, states: Sequence[str], *, batch_size: int = 128) -> torch.Tensor:
        return self._encode_texts(states, batch_size=batch_size)

    @torch.inference_mode()
    def score_states(
        self,
        states: Sequence[str],
        cached_options: CachedOptions,
        *,
        batch_size: int = 128,
    ) -> torch.Tensor:
        if cached_options.owner is not self._cache_owner:
            raise ValueError("cached options were encoded by a different model instance")
        state_vectors = self.encode_states(states, batch_size=batch_size)
        return self.model.scale() * state_vectors @ cached_options.vectors.to(self.device).T

    @torch.inference_mode()
    def decide(
        self,
        state: str,
        question: str,
        options: Sequence[str],
        *,
        cached_options: CachedOptions | None = None,
        temperature: float = 1.0,
        abstain_option: str | None = None,
        abstain_threshold: float | None = None,
        abstain_temperature: float | None = None,
        calibrated_options: Sequence[str] | None = None,
        calibrated_question: str | None = None,
        calibrated_max_length: int | None = None,
    ) -> ChoiceAnswer:
        if cached_options is None:
            cached_options = self.encode_options(question, options)
        cached_options.validate(self._cache_owner, question, options)
        if abstain_threshold is not None:
            if (
                calibrated_options is None
                or calibrated_question is None
                or calibrated_max_length is None
            ):
                raise ValueError("complete calibration metadata is required with an abstain threshold")
            if set(calibrated_options) != set(options):
                raise ValueError("runtime options do not match the calibrated option catalog")
            if calibrated_question != question:
                raise ValueError("runtime question does not match the calibrated question")
            if calibrated_max_length != self.max_length:
                raise ValueError("runtime max_length does not match the calibrated max_length")
        logits = self.score_states([state], cached_options)
        return ChoiceAnswer.from_logits(
            options,
            logits[0].detach().cpu().tolist(),
            temperature=temperature,
            abstain_option=abstain_option,
            abstain_threshold=abstain_threshold,
            abstain_temperature=abstain_temperature,
        )

    def save(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.encoder.save_pretrained(destination / "encoder")
        self.tokenizer.save_pretrained(destination)
        save_file(
            {
                "projection.weight": self.model.projection.weight.detach().cpu().contiguous(),
                "logit_scale": self.model.logit_scale.detach().cpu().reshape(()).contiguous(),
            },
            destination / BIENCODER_HEAD,
        )
        (destination / BIENCODER_CONFIG).write_text(
            json.dumps(
                {
                    "architecture": "shared-biencoder",
                    "projection_dim": self.model.projection.out_features,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
