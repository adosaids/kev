from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kev.biencoder import BiEncoderDecisionModel
from kev.data import NONE_OF_ABOVE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make one cached-option bi-encoder decision")
    parser.add_argument("--model", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--option", action="append", required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--rejection-threshold", type=float)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperature = args.temperature
    rejection_threshold = args.rejection_threshold
    calibration_path = Path(args.model) / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if temperature is None:
            temperature = calibration["temperature"]
        if rejection_threshold is None:
            rejection_threshold = calibration.get("rejection_threshold")
    if temperature is None:
        temperature = 1.0

    decision_model = BiEncoderDecisionModel.from_pretrained(
        args.model,
        max_length=args.max_length,
    )
    decision_model.model.to(args.device)
    options = list(args.option)
    if rejection_threshold is not None and NONE_OF_ABOVE not in options:
        options.append(NONE_OF_ABOVE)
    cached_options = decision_model.encode_options(args.question, options)
    answer = decision_model.decide(
        args.state,
        args.question,
        options,
        cached_options=cached_options,
        temperature=temperature,
        abstain_option=NONE_OF_ABOVE if rejection_threshold is not None else None,
        abstain_threshold=rejection_threshold,
    )
    print(
        json.dumps(
            {
                "type": "choice",
                "choice": answer.choice,
                "probabilities": answer.probabilities,
                "confidence": answer.confidence,
                "rejected": answer.rejected,
                "rejection_score": answer.rejection_score,
                "temperature": temperature,
                "rejection_threshold": rejection_threshold,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
