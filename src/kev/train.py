from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from kev.data import (
    ChoiceExample,
    grouped_candidates,
    load_categories,
    load_split,
    train_validation_split,
)
from kev.model import (
    DEFAULT_BASE_MODEL,
    DEFAULT_QUESTION,
    CandidatePair,
    CrossEncoderDecisionModel,
)


class TrainingDataset(Dataset):
    def __init__(
        self,
        examples: list[ChoiceExample],
        labels: list[str],
        *,
        negatives: int,
        seed: int,
    ):
        self.examples = examples
        self.labels = labels
        self.negatives = negatives
        self.seed = seed

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return grouped_candidates(
            self.examples[index],
            self.labels,
            negatives=self.negatives,
            seed=self.seed + index,
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_collator(decision_model: CrossEncoderDecisionModel):
    def collate(groups):
        pairs: list[CandidatePair] = []
        targets: list[int] = []
        group_size = len(groups[0].options)
        for group in groups:
            if len(group.options) != group_size:
                raise ValueError("all training groups must have the same size")
            pairs.extend(
                CandidatePair(group.state, group.question, option)
                for option in group.options
            )
            targets.append(group.target)
        encoded = decision_model.tokenize_pairs(pairs)
        return encoded, torch.tensor(targets), group_size

    return collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a grouped-choice cross encoder")
    parser.add_argument("--data-dir", default="data/banking77")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="artifacts/kev-minilm")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8, help="choice groups per step")
    parser.add_argument("--negatives", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    labels = load_categories(data_dir)
    examples, validation_examples = train_validation_split(
        load_split(data_dir, "train"),
        fraction=0.1,
        seed=args.seed,
    )
    if args.max_train_samples:
        examples = examples[: args.max_train_samples]

    decision_model = CrossEncoderDecisionModel.from_pretrained(
        args.base_model,
        max_length=args.max_length,
        trainable_head=True,
    )
    decision_model.model.to(args.device)
    dataset = TrainingDataset(examples, labels, negatives=args.negatives, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collator(decision_model),
        pin_memory=args.device.startswith("cuda"),
    )
    optimizer = torch.optim.AdamW(
        decision_model.model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    started = time.perf_counter()

    decision_model.model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_correct = 0
        seen = 0
        for step, (encoded, targets, group_size) in enumerate(loader, start=1):
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            targets = targets.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = decision_model.model(**encoded).logits.reshape(-1, group_size)
                loss = functional.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * len(targets)
            running_correct += int((logits.argmax(dim=-1) == targets).sum().item())
            seen += len(targets)
            if step % 50 == 0 or step == len(loader):
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "steps": len(loader),
                            "loss": running_loss / seen,
                            "group_accuracy": running_correct / seen,
                        }
                    )
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_model.save(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "question": DEFAULT_QUESTION,
                "labels": labels,
                "reserved_validation_samples": len(validation_examples),
                "training_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
