from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from kev.calibration import fit_temperature
from kev.data import ChoiceExample, load_categories, load_split, train_validation_split
from kev.metrics import classification_metrics, coverage_at_error
from kev.model import DEFAULT_QUESTION, CrossEncoderDecisionModel


def collect_logits(
    decision_model: CrossEncoderDecisionModel,
    examples: list[ChoiceExample],
    labels: list[str],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    rows: list[list[float]] = []
    targets: list[int] = []
    started = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        rows.append(
            decision_model.score_options(
                example.text,
                DEFAULT_QUESTION,
                labels,
                batch_size=batch_size,
            )
        )
        targets.append(labels.index(example.label))
        if index % 50 == 0:
            print(f"evaluated {index}/{len(examples)}")
    elapsed = time.perf_counter() - started
    return torch.tensor(rows), torch.tensor(targets), elapsed


def evaluate_logits(logits: torch.Tensor, targets: torch.Tensor, temperature: float):
    probabilities = torch.softmax(logits / temperature, dim=-1).tolist()
    labels = targets.tolist()
    return {
        **classification_metrics(probabilities, labels),
        "coverage_at_5pct_error": coverage_at_error(probabilities, labels, max_error=0.05),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and calibrate a kev checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="data/banking77")
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    labels = load_categories(args.data_dir)
    _, validation_examples = train_validation_split(
        load_split(args.data_dir, "train"),
        fraction=0.1,
        seed=args.seed,
    )
    test_examples = load_split(args.data_dir, "test")
    rng.shuffle(test_examples)
    calibration_examples = validation_examples[: args.calibration_samples]
    test_examples = test_examples[: args.test_samples]

    decision_model = CrossEncoderDecisionModel.from_pretrained(
        args.model,
        max_length=args.max_length,
    )
    decision_model.model.to(args.device)

    calibration_logits, calibration_targets, calibration_seconds = collect_logits(
        decision_model,
        calibration_examples,
        labels,
        batch_size=args.batch_size,
    )
    temperature = fit_temperature(calibration_logits, calibration_targets)
    test_logits, test_targets, test_seconds = collect_logits(
        decision_model,
        test_examples,
        labels,
        batch_size=args.batch_size,
    )

    original_predictions = test_logits.argmax(dim=-1)
    permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(args.seed))
    permuted_predictions = permutation[test_logits[:, permutation].argmax(dim=-1)]
    order_consistency = float((original_predictions == permuted_predictions).float().mean().item())

    result = {
        "model": args.model,
        "temperature": temperature,
        "calibration_samples": len(calibration_examples),
        "test_samples": len(test_examples),
        "uncalibrated": evaluate_logits(test_logits, test_targets, 1.0),
        "calibrated": evaluate_logits(test_logits, test_targets, temperature),
        "option_order_consistency": order_consistency,
        "calibration_seconds": calibration_seconds,
        "test_seconds": test_seconds,
        "milliseconds_per_decision": 1000 * test_seconds / max(len(test_examples), 1),
    }
    output_path = Path(args.model) / "evaluation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (Path(args.model) / "calibration.json").write_text(
        json.dumps({"temperature": temperature}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
