# Cached-option bi-encoder experiment — 2026-09-21

This experiment tests the proposed architecture where static question/option representations are
precomputed and each live state is encoded once.

## Controlled comparison

Both models used:

- `microsoft/MiniLM-L12-H384-uncased`
- 5,000 Banking77 training examples
- one epoch, batch size 8 and seven sampled negatives
- a stratified validation split reserved before training
- 200 calibration examples and the same 500 untouched test examples
- all 77 labels at evaluation time
- NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB

| Metric | Cross-encoder | Cached-option bi-encoder |
|---|---:|---:|
| Training time | 36.59 s | 45.25 s |
| Accuracy | 77.0% | **79.8%** |
| Calibrated Brier | 0.3368 | **0.2836** |
| Calibrated NLL | 0.8223 | **0.6580** |
| Calibrated ECE | **0.0425** | 0.0464 |
| Coverage at no more than 5% empirical error | 53.2% | **56.6%** |
| Online latency per 77-option decision | 31.58 ms | **13.77 ms** |
| Option-order consistency | 100% | 100% |

The bi-encoder was about **2.29x faster online** in this test. Encoding all 77 static options once
took 0.73 seconds; an application would keep those vectors in memory and amortize that cost across
requests. The crossover is roughly 41 decisions with these measured timings.

The result supports the architecture for a fixed or slowly changing tool catalog. It does not yet
test unseen option descriptions, out-of-domain rejection, Chinese input or concurrent serving.
The 500-example evaluation is also too small to establish a production error guarantee.

## Commands

```powershell
kev-train-bi `
  --data-dir data/banking77 `
  --output-dir artifacts/kev-bi-minilm-5k `
  --max-train-samples 5000 `
  --epochs 1 `
  --batch-size 8 `
  --negatives 7 `
  --max-length 96

kev-eval-bi `
  --model artifacts/kev-bi-minilm-5k `
  --data-dir data/banking77 `
  --calibration-samples 200 `
  --test-samples 500 `
  --max-length 96
```

