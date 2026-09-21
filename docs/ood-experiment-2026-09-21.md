# Explicit OOD rejection experiment — 2026-09-21

This experiment adds a typed `none_of_above` outcome to the cached-option bi-encoder. Known
Banking77 examples see it as a negative candidate. CLINC150 OOS examples use it as the positive
candidate. The output remains constrained to caller options plus this reserved abstention option.

## Data integrity

- Base checkpoint: `artifacts/kev-bi-minilm-5k`
- Known training: 5,000 Banking77 examples
- OOD training: 250 CLINC `oos_train` examples repeated four times
- Calibration: 1,000 reserved Banking77 examples plus all 100 CLINC `oos_val` examples
- Final test: all 3,080 Banking77 examples plus all 1,000 CLINC `oos_test` examples
- CLINC source: official `clinc/oos-eval` repository at commit
  `828f8093932c8fe6ca7936c3d2e52903b1c523de`

Classification temperature is fit only on known validation examples. Rejection uses
`sigmoid(none_logit - best_known_logit)` and a separate threshold that maximizes balanced
accept/reject accuracy on known and OOD validation data. The final test sets are not used to select
either value.

## RTX 4060 results

### Far-OOD: Banking77 versus CLINC

This full benchmark uses all 3,080 Banking77 test examples and all 1,000 CLINC OOS test examples.

| Metric | Far-OOD result |
|---|---:|
| Additional training time | 58.91 s |
| Known-intent accuracy | 78.99% |
| Known accuracy after rejection | 78.51% |
| Known false-reject rate | 1.10% |
| OOD recall | 97.7% |
| OOD false-accept rate | 2.3% |
| Rejection balanced accuracy | 98.30% |
| Open-set accuracy | 83.21% |
| Calibrated ECE on known test | 0.0499 |
| Rejection threshold | 0.021910 |
| Online latency per 78-option decision | 8.39–13.74 ms |
| Option-order consistency | 100% |

### Hard ID-OOS: 50 supported versus 27 held-out banking intents

The paper's BANKING77-OOS split keeps 50 intents as supported behaviors and holds out 27 entire
banking intents as near-domain OOD. A fresh model was trained from the original MiniLM checkpoint,
so it never trained on those 27 intent classes. CLINC OOS supplied only the generic abstention
training examples. The preparation script removes empty strings and exact cross-split duplicates
found in the published files. Threshold calibration uses 528 hard-OOD validation examples; final
evaluation uses 1,999 known and 1,072 hard-OOD test examples.

| Metric | Hard ID-OOS result |
|---|---:|
| Additional training time | 66.61 s |
| Known-intent accuracy | 72.39% |
| Known accuracy after rejection | 51.38% |
| Known false-reject rate | 32.32% |
| OOD recall | 62.78% |
| OOD false-accept rate | 37.22% |
| Rejection balanced accuracy | 65.23% |
| Open-set accuracy | 55.36% |
| Calibrated ECE on known test | 0.1105 |
| Online latency per 51-option decision | 8.21–8.31 ms |
| Option-order consistency | 100% |

The far-OOD result is real for that distribution, but it does not transfer to semantically similar
unsupported requests. The hard split exposes the current model's main weakness: a single learned
`none_of_above` vector does not form a reliable boundary around many fine-grained supported intents.
Adversarial phrasing, Chinese input and distribution drift remain untested.

## Commands

```powershell
kev-download-oos --output data/clinc150

git clone https://github.com/jianguoz/Few-Shot-Intent-Detection `
  .tmp-datasets/Few-Shot-Intent-Detection
git -C .tmp-datasets/Few-Shot-Intent-Detection checkout `
  fa9553312b1dba54b1be0aae2d30b13171f898e5
python scripts/prepare_banking77_oos.py `
  --source .tmp-datasets/Few-Shot-Intent-Detection/Datasets/BANKING77-OOS

kev-train-bi `
  --data-dir data/banking77 `
  --ood-data-dir data/clinc150 `
  --base-model artifacts/kev-bi-minilm-5k `
  --output-dir artifacts/kev-bi-minilm-ood-5k `
  --max-train-samples 5000 `
  --ood-repeats 4 `
  --epochs 1 `
  --batch-size 8 `
  --negatives 7 `
  --max-length 96

kev-eval-bi `
  --model artifacts/kev-bi-minilm-ood-5k `
  --data-dir data/banking77 `
  --ood-data-dir data/clinc150 `
  --calibration-samples 1000 `
  --ood-calibration-samples 100 `
  --test-samples 3080 `
  --ood-test-samples 1000 `
  --max-length 96

kev-train-bi `
  --data-dir data/banking77-oos-50 `
  --ood-data-dir data/banking77-hard-oos `
  --base-model microsoft/MiniLM-L12-H384-uncased `
  --output-dir artifacts/kev-bi-hard-oos-50-clean `
  --max-train-samples 5000 `
  --ood-repeats 4 `
  --epochs 1 `
  --batch-size 8 `
  --negatives 7 `
  --max-length 96

kev-eval-bi `
  --model artifacts/kev-bi-hard-oos-50-clean `
  --data-dir data/banking77-oos-50 `
  --ood-data-dir data/banking77-hard-oos `
  --calibration-samples 500 `
  --ood-calibration-samples 528 `
  --test-samples 1999 `
  --ood-test-samples 1072 `
  --max-length 96
```
