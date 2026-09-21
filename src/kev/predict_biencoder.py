from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kev.biencoder import BiEncoderDecisionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make one cached-option bi-encoder decision")
    parser.add_argument("--model", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--option", action="append", required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperature = args.temperature
    calibration_path = Path(args.model) / "calibration.json"
    if temperature is None and calibration_path.exists():
        temperature = json.loads(calibration_path.read_text(encoding="utf-8"))["temperature"]
    if temperature is None:
        temperature = 1.0

    decision_model = BiEncoderDecisionModel.from_pretrained(
        args.model,
        max_length=args.max_length,
    )
    decision_model.model.to(args.device)
    option_vectors = decision_model.encode_options(args.question, args.option)
    answer = decision_model.decide(
        args.state,
        args.question,
        args.option,
        option_vectors=option_vectors,
        temperature=temperature,
    )
    print(
        json.dumps(
            {
                "type": "choice",
                "choice": answer.choice,
                "probabilities": answer.probabilities,
                "confidence": answer.confidence,
                "temperature": temperature,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

