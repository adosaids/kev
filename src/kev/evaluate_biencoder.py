from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from kev.biencoder import BiEncoderDecisionModel, CachedOptions
from kev.calibration import fit_temperature
from kev.data import ChoiceExample, load_categories, load_split, train_validation_split
from kev.evaluate import evaluate_logits, validate_sample_counts
from kev.model import DEFAULT_QUESTION


@torch.inference_mode()
def collect_logits(
    decision_model: BiEncoderDecisionModel,
    examples: list[ChoiceExample],
    *,
    cached_options: CachedOptions,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    rows: list[torch.Tensor] = []
    targets: list[int] = []
    started = time.perf_counter()
    for example in examples:
        rows.append(decision_model.score_states([example.text], cached_options).cpu()[0])
        targets.append(cached_options.options.index(example.label))
    elapsed = time.perf_counter() - started
    return torch.stack(rows), torch.tensor(targets), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the cached-option bi-encoder")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", default="data/banking77")
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    validate_sample_counts(args.calibration_samples, args.test_samples)
    rng = random.Random(args.seed)
    labels = load_categories(args.data_dir)
    _, validation_examples = train_validation_split(
        load_split(args.data_dir, "train"), fraction=0.1, seed=args.seed
    )
    test_examples = load_split(args.data_dir, "test")
    rng.shuffle(test_examples)
    validation_examples = validation_examples[: args.calibration_samples]
    test_examples = test_examples[: args.test_samples]

    decision_model = BiEncoderDecisionModel.from_pretrained(
        args.model,
        max_length=args.max_length,
    )
    decision_model.model.to(args.device)

    synchronize_if_cuda(decision_model.device)
    option_started = time.perf_counter()
    cached_options = decision_model.encode_options(DEFAULT_QUESTION, labels)
    synchronize_if_cuda(decision_model.device)
    option_precompute_seconds = time.perf_counter() - option_started
    calibration_logits, calibration_targets, calibration_seconds = collect_logits(
        decision_model,
        validation_examples,
        cached_options=cached_options,
    )
    temperature = fit_temperature(calibration_logits, calibration_targets)
    test_logits, test_targets, test_seconds = collect_logits(
        decision_model,
        test_examples,
        cached_options=cached_options,
    )

    permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(args.seed))
    permuted_labels = [labels[index] for index in permutation.tolist()]
    permuted_options = decision_model.encode_options(DEFAULT_QUESTION, permuted_labels)
    permuted_logits, _, order_seconds = collect_logits(
        decision_model,
        test_examples,
        cached_options=permuted_options,
    )
    original_predictions = [labels[index] for index in test_logits.argmax(dim=-1).tolist()]
    permuted_predictions = [
        permuted_labels[index] for index in permuted_logits.argmax(dim=-1).tolist()
    ]
    order_consistency = sum(
        original == permuted
        for original, permuted in zip(original_predictions, permuted_predictions, strict=True)
    ) / len(original_predictions)

    result = {
        "model": args.model,
        "temperature": temperature,
        "calibration_samples": len(validation_examples),
        "test_samples": len(test_examples),
        "uncalibrated": evaluate_logits(test_logits, test_targets, 1.0),
        "calibrated": evaluate_logits(test_logits, test_targets, temperature),
        "option_order_consistency": order_consistency,
        "option_precompute_seconds": option_precompute_seconds,
        "order_consistency_seconds": order_seconds,
        "calibration_seconds": calibration_seconds,
        "test_seconds": test_seconds,
        "milliseconds_per_online_decision": 1000 * test_seconds / len(test_examples),
    }
    output_path = Path(args.model) / "evaluation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (Path(args.model) / "calibration.json").write_text(
        json.dumps({"temperature": temperature}, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
