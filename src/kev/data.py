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
CLINC_OOS_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_OOS_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/"
    f"{CLINC_OOS_COMMIT}/data/data_oos_plus.json"
)
NONE_OF_ABOVE = "none_of_above"


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


def abstaining_grouped_candidates(
    example: ChoiceExample,
    labels: Sequence[str],
    *,
    negatives: int,
    seed: int,
    question: str = "What is the customer's banking intent?",
) -> CandidateGroup:
    if NONE_OF_ABOVE in labels:
        raise ValueError("labels must not contain the reserved none_of_above option")
    if negatives < 1:
        raise ValueError("abstention training requires at least one negative")

    rng = random.Random(seed)
    if example.label == NONE_OF_ABOVE:
        if negatives > len(labels):
            raise ValueError("negatives is outside the available candidate range")
        options = [NONE_OF_ABOVE, *rng.sample(list(labels), negatives)]
    else:
        if example.label not in labels:
            raise ValueError(f"unknown label: {example.label}")
        available = [label for label in labels if label != example.label]
        if negatives - 1 > len(available):
            raise ValueError("negatives is outside the available candidate range")
        options = [example.label, NONE_OF_ABOVE, *rng.sample(available, negatives - 1)]
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


def download_clinc_oos(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CLINC_OOS_URL, destination / "data_oos_plus.json")
    (destination / "SOURCE.txt").write_text(
        "CLINC150 OOS data from clinc/oos-eval\n"
        f"Commit: {CLINC_OOS_COMMIT}\n"
        "License: CC BY 4.0\n",
        encoding="utf-8",
    )
    return destination


def load_clinc_oos(data_dir: str | Path, split: str) -> list[ChoiceExample]:
    if split not in {"train", "val", "test"}:
        raise ValueError("CLINC OOS split must be train, val or test")
    path = Path(data_dir) / "data_oos_plus.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ChoiceExample(text, NONE_OF_ABOVE) for text, _ in payload[f"oos_{split}"]]


def download_cli() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned Banking77 dataset")
    parser.add_argument("--output", default="data/banking77")
    args = parser.parse_args()
    print(download_banking77(args.output))


def download_oos_cli() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned CLINC150 OOS dataset")
    parser.add_argument("--output", default="data/clinc150")
    args = parser.parse_args()
    print(download_clinc_oos(args.output))


if __name__ == "__main__":
    download_cli()
