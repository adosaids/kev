from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from kev.biencoder import BiEncoderDecisionModel, CachedOptions
from kev.calibration import fit_temperature
from kev.data import (
    NONE_OF_ABOVE,
    ChoiceExample,
    load_categories,
    load_clinc_oos,
    load_split,
    train_validation_split,
)
from kev.evaluate import evaluate_logits, validate_sample_counts
from kev.model import DEFAULT_QUESTION
from kev.rejection import fit_rejection_threshold, rejection_metrics


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
    parser.add_argument("--ood-data-dir", default="data/clinc150")
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--ood-calibration-samples", type=int, default=100)
    parser.add_argument("--ood-test-samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rejection_scores(logits: torch.Tensor, *, none_index: int, temperature: float) -> list[float]:
    scaled = logits / temperature
    known_columns = [index for index in range(logits.shape[1]) if index != none_index]
    best_known = scaled[:, known_columns].max(dim=-1).values
    return torch.sigmoid(scaled[:, none_index] - best_known).tolist()


def main() -> None:
    args = parse_args()
    validate_sample_counts(args.calibration_samples, args.test_samples)
    validate_sample_counts(args.ood_calibration_samples, args.ood_test_samples)
    rng = random.Random(args.seed)
    labels = load_categories(args.data_dir)
    _, validation_examples = train_validation_split(
        load_split(args.data_dir, "train"), fraction=0.1, seed=args.seed
    )
    test_examples = load_split(args.data_dir, "test")
    ood_calibration_examples = load_clinc_oos(args.ood_data_dir, "val")
    ood_test_examples = load_clinc_oos(args.ood_data_dir, "test")
    rng.shuffle(test_examples)
    rng.shuffle(ood_calibration_examples)
    rng.shuffle(ood_test_examples)
    validation_examples = validation_examples[: args.calibration_samples]
    test_examples = test_examples[: args.test_samples]
    ood_calibration_examples = ood_calibration_examples[: args.ood_calibration_samples]
    ood_test_examples = ood_test_examples[: args.ood_test_samples]

    decision_model = BiEncoderDecisionModel.from_pretrained(
        args.model,
        max_length=args.max_length,
    )
    decision_model.model.to(args.device)

    synchronize_if_cuda(decision_model.device)
    option_started = time.perf_counter()
    open_set_labels = [*labels, NONE_OF_ABOVE]
    cached_options = decision_model.encode_options(DEFAULT_QUESTION, open_set_labels)
    synchronize_if_cuda(decision_model.device)
    option_precompute_seconds = time.perf_counter() - option_started
    calibration_logits, calibration_targets, calibration_seconds = collect_logits(
        decision_model,
        validation_examples,
        cached_options=cached_options,
    )
    ood_calibration_logits, ood_calibration_targets, ood_calibration_seconds = collect_logits(
        decision_model,
        ood_calibration_examples,
        cached_options=cached_options,
    )
    combined_calibration_logits = torch.cat([calibration_logits, ood_calibration_logits])
    combined_calibration_targets = torch.cat([calibration_targets, ood_calibration_targets])
    temperature = fit_temperature(combined_calibration_logits, combined_calibration_targets)
    none_index = cached_options.options.index(NONE_OF_ABOVE)
    known_calibration_none_scores = rejection_scores(
        calibration_logits, none_index=none_index, temperature=temperature
    )
    ood_calibration_none_scores = rejection_scores(
        ood_calibration_logits, none_index=none_index, temperature=temperature
    )
    rejection_threshold = fit_rejection_threshold(
        known_calibration_none_scores,
        ood_calibration_none_scores,
    )
    test_logits, test_targets, test_seconds = collect_logits(
        decision_model,
        test_examples,
        cached_options=cached_options,
    )
    ood_test_logits, _, ood_test_seconds = collect_logits(
        decision_model,
        ood_test_examples,
        cached_options=cached_options,
    )

    test_probabilities = torch.softmax(test_logits / temperature, dim=-1)
    known_none_scores = rejection_scores(
        test_logits, none_index=none_index, temperature=temperature
    )
    ood_none_scores = rejection_scores(
        ood_test_logits, none_index=none_index, temperature=temperature
    )
    rejection = rejection_metrics(
        known_none_scores,
        ood_none_scores,
        threshold=rejection_threshold,
    )
    known_predictions = test_probabilities[:, : len(labels)].argmax(dim=-1)
    known_accepted = torch.tensor(known_none_scores) < rejection_threshold
    known_correct = known_predictions == test_targets
    in_scope_intent_accuracy = float(known_correct.float().mean().item())
    in_scope_accuracy_with_rejection = float(
        (known_correct & known_accepted).float().mean().item()
    )
    ood_rejected = torch.tensor(ood_none_scores) >= rejection_threshold
    open_set_accuracy = float(
        (
            (known_correct & known_accepted).sum() + ood_rejected.sum()
        ).item()
        / (len(test_examples) + len(ood_test_examples))
    )

    permutation = torch.randperm(
        len(open_set_labels), generator=torch.Generator().manual_seed(args.seed)
    )
    permuted_labels = [open_set_labels[index] for index in permutation.tolist()]
    permuted_options = decision_model.encode_options(DEFAULT_QUESTION, permuted_labels)
    permuted_logits, _, order_seconds = collect_logits(
        decision_model,
        test_examples,
        cached_options=permuted_options,
    )
    original_predictions = [
        open_set_labels[index] for index in test_logits.argmax(dim=-1).tolist()
    ]
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
        "ood_calibration_samples": len(ood_calibration_examples),
        "ood_test_samples": len(ood_test_examples),
        "uncalibrated": evaluate_logits(test_logits, test_targets, 1.0),
        "calibrated": evaluate_logits(test_logits, test_targets, temperature),
        "rejection": rejection,
        "in_scope_intent_accuracy": in_scope_intent_accuracy,
        "in_scope_accuracy_with_rejection": in_scope_accuracy_with_rejection,
        "open_set_accuracy": open_set_accuracy,
        "option_order_consistency": order_consistency,
        "option_precompute_seconds": option_precompute_seconds,
        "order_consistency_seconds": order_seconds,
        "calibration_seconds": calibration_seconds,
        "ood_calibration_seconds": ood_calibration_seconds,
        "test_seconds": test_seconds,
        "ood_test_seconds": ood_test_seconds,
        "milliseconds_per_online_decision": 1000 * test_seconds / len(test_examples),
    }
    output_path = Path(args.model) / "evaluation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (Path(args.model) / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "rejection_threshold": rejection_threshold,
                "abstain_option": NONE_OF_ABOVE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
