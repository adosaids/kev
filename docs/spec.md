# kev v0 specification

Build a small Jev-style typed decision model that can be trained on an 8 GB RTX 4060.

## Public seams

1. **Data**: download Banking77 from its pinned public GitHub revision and expose examples as
   `(state, question, candidate options, correct option)`.
2. **Training CLI**: fine-tune a small cross-encoder with grouped choice loss and save a standard
   Transformers checkpoint.
3. **Inference and evaluation CLI**: return only caller-declared choices, full probabilities and
   confidence; report accuracy, probability-quality, selective-risk, latency and option-order
   consistency metrics.

## Constraints

- Run on an 8 GB NVIDIA RTX 4060 Laptop GPU.
- Do not generate free-form answers: map an option index back to a caller-provided option.
- Use softmax for mutually exclusive Choice decisions.
- Fit post-hoc temperature scaling on validation data, never on the test split.
- Make small smoke runs possible before committing to a full training run.
- Keep downloaded data and trained weights out of Git.

