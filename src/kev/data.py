from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


BANKING77_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{BANKING77_COMMIT}/banking_data"
)


@dataclass(frozen=True)
class ChoiceExample:
    text: str
    label: str


@dataclass(frozen=True)
class CandidateGroup:
    state: str
    question: str
    options: tuple[str, ...]
    target: int


def option_description(label: str) -> str:
    return label.replace("_", " ")


def grouped_candidates(
    example: ChoiceExample,
    labels: Sequence[str],
    *,
    negatives: int,
    seed: int,
    question: str = "What is the customer's banking intent?",
) -> CandidateGroup:
    if example.label not in labels:
        raise ValueError(f"unknown label: {example.label}")
    available = [label for label in labels if label != example.label]
    if not 0 <= negatives <= len(available):
        raise ValueError("negatives is outside the available candidate range")

    rng = random.Random(seed)
    options = [example.label, *rng.sample(available, negatives)]
    rng.shuffle(options)
    return CandidateGroup(
        state=example.text,
        question=question,
        options=tuple(options),
        target=options.index(example.label),
    )


def load_categories(data_dir: str | Path) -> list[str]:
    with (Path(data_dir) / "categories.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_split(data_dir: str | Path, split: str) -> list[ChoiceExample]:
    path = Path(data_dir) / f"{split}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return [ChoiceExample(row["text"], row["category"]) for row in csv.DictReader(handle)]


def train_validation_split(
    examples: Sequence[ChoiceExample],
    *,
    fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[ChoiceExample], list[ChoiceExample]]:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    by_label: dict[str, list[ChoiceExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)

    train: list[ChoiceExample] = []
    validation: list[ChoiceExample] = []
    for label in sorted(by_label):
        group = list(by_label[label])
        random.Random(f"{seed}:{label}").shuffle(group)
        validation_size = max(1, round(len(group) * fraction))
        validation.extend(group[:validation_size])
        train.extend(group[validation_size:])
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    return train, validation


def download_banking77(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("train.csv", "test.csv", "categories.json"):
        urllib.request.urlretrieve(f"{BANKING77_BASE_URL}/{filename}", destination / filename)
    (destination / "SOURCE.txt").write_text(
        "Banking77 from PolyAI-LDN/task-specific-datasets\n"
        f"Commit: {BANKING77_COMMIT}\n"
        "License: CC BY 4.0\n",
        encoding="utf-8",
    )
    return destination


def download_cli() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned Banking77 dataset")
    parser.add_argument("--output", default="data/banking77")
    args = parser.parse_args()
    print(download_banking77(args.output))


if __name__ == "__main__":
    download_cli()
