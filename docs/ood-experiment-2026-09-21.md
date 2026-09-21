# Explicit OOD rejection experiment — 2026-09-21

This experiment adds a typed `none_of_above` outcome to the cached-option bi-encoder. Known
Banking77 examples see it as a negative candidate. CLINC150 OOS examples use it as the positive
candidate. The output remains constrained to caller options plus this reserved abstention option.

## Data integrity

- Base checkpoint: `artifacts/kev-bi-minilm-5k`
- Known training: 5,000 Banking77 examples
- OOD training: 250 CLINC `oos_train` examples repeated four times
- Calibration: 200 reserved Banking77 examples plus all 100 CLINC `oos_val` examples
- Final test: 500 untouched Banking77 examples plus 500 CLINC `oos_test` examples
- CLINC source: official `clinc/oos-eval` repository at commit
  `828f8093932c8fe6ca7936c3d2e52903b1c523de`

Temperature is fit on the combined validation logits. Rejection uses
`sigmoid((none_logit - best_known_logit) / temperature)`. Its threshold maximizes balanced
accept/reject accuracy on validation data. The final test sets are not used to select either value.

## RTX 4060 result

| Metric | Result |
|---|---:|
| Additional training time | 58.91 s |
| Known-intent accuracy | 83.2% |
| Known accuracy after rejection | 83.2% |
| Known false-reject rate | 0.0% |
| OOD recall | 97.4% |
| OOD false-accept rate | 2.6% |
| Rejection balanced accuracy | 98.7% |
| Open-set accuracy | 90.3% |
| Calibrated ECE on known test | 0.0614 |
| Rejection threshold | 0.005486 |
| Online latency per 78-option decision | 7.96 ms |
| Option-order consistency | 100% |

The model correctly accepted `I am still waiting for my card` as `card_arrival` and rejected
`Who won the football match yesterday?` as `none_of_above` when given only three real options.

This is an easy cross-domain OOD setting: Banking77 is narrowly financial while much of CLINC OOS
is clearly unrelated. The 0% observed false-reject rate is therefore not a production guarantee.
Near-domain banking requests for unsupported products, adversarial phrasing, Chinese input and
distribution drift still need separate hard-OOD test sets.

## Commands

```powershell
kev-download-oos --output data/clinc150

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
  --calibration-samples 200 `
  --ood-calibration-samples 100 `
  --test-samples 500 `
  --ood-test-samples 500 `
  --max-length 96
```
