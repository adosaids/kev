# kev

`kev` is a small Jev-style typed decision model experiment designed to train on an 8 GB
consumer GPU. It scores choices supplied by the caller and maps the winning index back to those
choices. It never generates an answer string, so an undeclared option is structurally impossible.

This repository is an experiment, not a reproduction of TypeSafe Jev. Jev's model architecture,
weights, training data and RLCD recipe are not public.

## What this version tests

- A small MiniLM cross-encoder scores `(state, question, candidate option)` triples.
- Grouped softmax trains one correct option against sampled negative options.
- Full-choice evaluation measures accuracy, multiclass Brier score, NLL and ECE.
- Temperature scaling is fit on held-out training examples, never on the test set.
- `coverage_at_5pct_error` measures how much work can be automated at a 5% empirical error budget.
- Inference returns a typed Choice object containing only caller-declared options.

The first dataset is Banking77 from
[PolyAI's task-specific datasets](https://github.com/PolyAI-LDN/task-specific-datasets), pinned to
commit `57ec275d8078af65b7731c2a98be812d844a6d6b` and distributed under CC BY 4.0. Downloaded data
and trained checkpoints are intentionally excluded from Git.

## Setup

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended.

```powershell
uv sync --extra dev
uv run kev-download --output data/banking77
```

If PyTorch is already installed for your CUDA version, install without replacing it:

```powershell
python -m pip wheel . --no-deps --wheel-dir dist
python -m pip install --force-reinstall --no-deps (Get-ChildItem dist/kev_decision-*.whl).FullName
kev-download --output data/banking77
```

On Windows, editable installs from a path containing non-ASCII characters can produce a broken
`.pth` mapping with some Python packaging combinations. Installing the built wheel avoids that
failure.

## Smoke training on an RTX 4060 8 GB

```powershell
kev-train `
  --data-dir data/banking77 `
  --output-dir artifacts/kev-minilm-smoke `
  --max-train-samples 1000 `
  --epochs 1 `
  --batch-size 8 `
  --negatives 4 `
  --max-length 96
```

For the full Banking77 training split, remove `--max-train-samples`. Start with 7 sampled
negatives. Evaluating all 77 choices is deliberately harder than evaluating only sampled groups.

## Calibrate and evaluate

```powershell
kev-eval `
  --model artifacts/kev-minilm-smoke `
  --data-dir data/banking77 `
  --calibration-samples 200 `
  --test-samples 500
```

This writes `calibration.json` and `evaluation.json` beside the checkpoint.

## Make a typed decision

```powershell
kev-predict `
  --model artifacts/kev-minilm-smoke `
  --state "I am still waiting for my card" `
  --question "What is the customer's banking intent?" `
  --option card_arrival `
  --option cash_withdrawal `
  --option exchange_rate
```

The output is always shaped like:

```json
{
  "type": "choice",
  "choice": "card_arrival",
  "probabilities": {
    "card_arrival": 0.91,
    "cash_withdrawal": 0.04,
    "exchange_rate": 0.05
  },
  "confidence": 0.91,
  "temperature": 1.4
}
```

The closed output contract prevents format hallucinations. It does not guarantee that the selected
option is correct. Production actions still need confidence gates, deterministic policy checks and
an abstain or escalation path.

## Cached-option bi-encoder

The second experiment pre-encodes static question/option descriptions and encodes each live state
once. Train and evaluate it with:

```powershell
kev-train-bi `
  --data-dir data/banking77 `
  --output-dir artifacts/kev-bi-minilm-5k `
  --max-train-samples 5000 `
  --epochs 1 `
  --batch-size 8 `
  --negatives 7

kev-eval-bi `
  --model artifacts/kev-bi-minilm-5k `
  --data-dir data/banking77 `
  --calibration-samples 200 `
  --test-samples 500
```

Applications should call `encode_options` once and reuse the returned identity-checked cache. The CLI demonstrates
the same closed output contract:

```powershell
kev-predict-bi `
  --model artifacts/kev-bi-minilm-5k `
  --state "I am still waiting for my card" `
  --question "What is the customer's banking intent?" `
  --option card_arrival `
  --option cash_withdrawal `
  --option exchange_rate
```

## Tests

```powershell
pytest -q
ruff check .
```

See [the experiment specification](docs/spec.md) and
[the architecture notes](docs/architecture.md) for scope and next steps. A real RTX 4060 run and
its limitations are recorded in [the cross-encoder report](docs/experiment-2026-09-21.md). The
[cached-option comparison](docs/biencoder-experiment-2026-09-21.md) records the faster architecture.
