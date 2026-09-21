# Bi-encoder experiment specification

Test whether a small decision model can pre-encode static questions and options, encode each live
state once, and retain useful choice accuracy while materially reducing online latency.

## Public seams

1. `BiEncoderDecisionModel.encode_options(question, options)` returns a reusable cache that binds
   vectors to the exact question and ordered option identities.
2. `BiEncoderDecisionModel.decide(state, question, options, cached_options=...)` returns the same
   closed `ChoiceAnswer` contract as the cross-encoder.
3. `kev-train-bi` trains grouped choices on the same reserved Banking77 split.
4. `kev-eval-bi` evaluates all 77 choices, validation-only temperature scaling, probability
   metrics, option-order consistency, option precomputation cost and online decision latency.
5. OOD-aware training adds a reserved `none_of_above` option. The rejection score is the sigmoid
   of its logit margin over the best real option; a validation-only threshold maps that
   score to accept or reject.
   The threshold is bound to the complete calibrated option catalog, question template and maximum
   token length, and must not be reused with different inputs.

## Constraints

- Fit on the existing 8 GB RTX 4060.
- Use attention-mask-aware mean pooling rather than treating `[SEP]` as a pooling token.
- L2-normalize state and option vectors, then learn a positive logit scale.
- Keep the downloaded data and generated checkpoints out of Git.
- Keep CLINC OOS train, validation and test splits separate. Fit temperature and rejection threshold
  on validation only, and report final OOD metrics on test only.
- Compare against the recorded cross-encoder experiment without rewriting its results.
