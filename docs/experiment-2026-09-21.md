# RTX 4060 experiment — 2026-09-21

This is a development experiment, not a full Banking77 benchmark.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- PyTorch: 2.6.0 + CUDA 12.4
- Transformers: 4.49.0
- Base model: `microsoft/MiniLM-L12-H384-uncased`
- Dataset: Banking77 at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`

## Training

```powershell
kev-train `
  --data-dir data/banking77 `
  --output-dir artifacts/kev-minilm-5k `
  --max-train-samples 5000 `
  --epochs 1 `
  --batch-size 8 `
  --negatives 7 `
  --max-length 96
```

- Training time: 36.59 seconds
- Sampled-group training accuracy: 81.94%
- A deterministic stratified 10% of the original training split was reserved before selecting the
  5,000 training examples.

## Evaluation

```powershell
kev-eval `
  --model artifacts/kev-minilm-5k `
  --data-dir data/banking77 `
  --calibration-samples 200 `
  --test-samples 500 `
  --batch-size 77 `
  --max-length 96
```

Each test example was scored against all 77 labels. Temperature was fitted on 200 reserved
validation examples, while the reported metrics use 500 examples from the untouched Banking77
test split.

| Metric | Raw | Temperature-scaled |
|---|---:|---:|
| Accuracy | 77.0% | 77.0% |
| Multiclass Brier | 0.3673 | 0.3368 |
| NLL | 0.9374 | 0.8223 |
| ECE | 0.1491 | 0.0425 |
| Coverage at no more than 5% empirical error | 48.0% | 53.2% |

- Fitted temperature: `0.7241`
- Option-order consistency: 100% in a second inference pass with a fixed random permutation of all
  77 options. Independent candidate scoring makes this expected, but the rerun verifies the public
  inference path rather than deriving it algebraically from the first pass.
- End-to-end latency: 31.84 ms per 77-option decision on this machine.

These numbers show that the small model learned useful zero-shot-style candidate scoring and that
post-hoc calibration materially improved probability quality. They do not establish production
reliability: the sample is small, the threshold was selected on the reported test slice, and the
model has no explicit out-of-domain or `none_of_the_above` training yet.
