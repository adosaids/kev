from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "fa9553312b1dba54b1be0aae2d30b13171f898e5"


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def unique_records(
    texts: list[str], labels: list[str], seen: set[str], *, split: str
) -> list[tuple[str, str]]:
    if len(texts) != len(labels):
        raise ValueError(f"mismatched files in {split}")
    records: list[tuple[str, str]] = []
    for text, label in zip(texts, labels, strict=True):
        normalized = text.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        records.append((normalized, label))
    return records


def write_known_split(destination: Path, split: str, records: list[tuple[str, str]]) -> None:
    with (destination / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows({"text": text, "category": label} for text, label in records)


def verify_source(source: Path) -> None:
    repository = source.parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"expected BANKING77-OOS commit {EXPECTED_COMMIT}, found {commit}")
    for arguments in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
        if subprocess.run(arguments, cwd=repository, check=False).returncode != 0:
            raise ValueError("BANKING77-OOS source repository has uncommitted changes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the BANKING77-OOS hard benchmark")
    parser.add_argument("--source", required=True)
    parser.add_argument("--clinc", default="data/clinc150/data_oos_plus.json")
    parser.add_argument("--known-output", default="data/banking77-oos-50")
    parser.add_argument("--ood-output", default="data/banking77-hard-oos")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    verify_source(source)
    known_output = Path(args.known_output)
    ood_output = Path(args.ood_output)
    known_output.mkdir(parents=True, exist_ok=True)
    ood_output.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    known: dict[str, list[tuple[str, str]]] = {}
    for split in ("train", "test"):
        known[split] = unique_records(
            read_lines(source / split / "seq.in"),
            read_lines(source / split / "label"),
            seen,
            split=f"known {split}",
        )
        write_known_split(known_output, split, known[split])
    labels = list(dict.fromkeys(label for _, label in known["train"]))
    (known_output / "categories.json").write_text(
        json.dumps(labels, indent=2), encoding="utf-8"
    )

    clinc = json.loads(Path(args.clinc).read_text(encoding="utf-8"))
    oos_train = unique_records(
        [row[0] for row in clinc["oos_train"]],
        ["oos"] * len(clinc["oos_train"]),
        seen,
        split="generic OOD train",
    )
    hard: dict[str, list[tuple[str, str]]] = {}
    for split in ("valid", "test"):
        texts = read_lines(source / "id-oos" / split / "seq.in")
        hard[split] = unique_records(
            texts, ["oos"] * len(texts), seen, split=f"hard OOD {split}"
        )
    payload = {
        "oos_train": [list(record) for record in oos_train],
        "oos_val": [list(record) for record in hard["valid"]],
        "oos_test": [list(record) for record in hard["test"]],
    }
    (ood_output / "data_oos_plus.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    source_note = (
        "BANKING77-OOS split from jianguoz/Few-Shot-Intent-Detection\n"
        f"Commit: {EXPECTED_COMMIT}\n"
        "Derived from Banking77 (CC BY 4.0); upstream split repository has no top-level license.\n"
        "Empty and cross-split duplicate texts removed by scripts/prepare_banking77_oos.py.\n"
    )
    for destination in (known_output, ood_output):
        (destination / "SOURCE.txt").write_text(source_note, encoding="utf-8")
    print(
        {
            "known_train": len(known["train"]),
            "known_test": len(known["test"]),
            "generic_ood_train": len(oos_train),
            "hard_ood_val": len(hard["valid"]),
            "hard_ood_test": len(hard["test"]),
        }
    )


if __name__ == "__main__":
    main()
