# Dynamic-option LLM architecture

## Decision

The next KEV experiment should use one small pretrained decoder-only LLM to read the state,
question and complete option set in a single context. A dynamic option head scores repeated option
boundary tokens. The model returns an option index that application code maps back to the exact
caller-provided key.

This design is intended to preserve more of an LLM's instruction and zero-shot generalization than
the current bi-encoder. It does not claim that the selected option is always correct. The structural
guarantee is narrower: the returned value must belong to the supplied option set.

## Input contract

Callers provide stable keys and meaningful natural-language descriptions:

```text
<STATE>
The customer says their physical card still has not arrived.
</STATE>
<QUESTION>
Which action should handle this request?
</QUESTION>
<OPTIONS>
<OPTION key="card_arrival">Track or investigate delivery of a physical card.</OPTION>
<OPTION key="cash_withdrawal">Handle an ATM cash withdrawal problem.</OPTION>
<OPTION key="exchange_rate">Explain a currency exchange rate.</OPTION>
<OPTION key="none_of_above">None of the supported actions applies.</OPTION>
</OPTIONS>
<DECIDE>
```

Keys are never treated as the semantic description. Training randomizes keys and option order so
the model must read the option text rather than memorize positions or label names. Inputs with
duplicate keys, an empty option list or multiple reserved abstention keys are rejected before
tokenization.

## Model

```text
state + question + every option
                |
         one decoder-only LLM
                |
    hidden state at final <DECIDE> token
           /                  \
option boundary states     optional answerability head
           |
 shared dynamic option scorer
           |
 one logit per runtime option
           |
      masked softmax
           |
 exact caller-provided option key
```

Add three special tokens to the tokenizer: `<OPTION>`, `</OPTION>` and `<DECIDE>`. For option `i`,
let `h_i` be the final hidden state at its `</OPTION>` token and let `h_d` be the hidden state at
`<DECIDE>`. Produce one score with a shared head:

```text
q   = normalize(Wq h_d)
o_i = normalize(Wo h_i)
z_i = exp(logit_scale) * dot(q, o_i)
p   = softmax(z / temperature)
```

The head has no fixed class dimension. The number of logits is the number of option boundary
tokens in that request. Every option representation sees the state, question and its own text;
`h_d` sees the complete option set. This keeps the pretrained LLM's token-level interaction while
still producing a typed dynamic choice.

Because a causal model can introduce option-order effects, training must randomize order and add an
order-consistency loss between two permutations of the same example. Evaluation must map indices
back to keys before measuring consistency.

## Output and abstention

The primary output is a structured value, not free text:

```json
{
  "type": "choice",
  "choice": "card_arrival",
  "probabilities": {
    "card_arrival": 0.82,
    "cash_withdrawal": 0.01,
    "exchange_rate": 0.02,
    "none_of_above": 0.15
  }
}
```

Two decoding implementations are acceptable:

1. Use the dynamic option logits directly and return `argmax` after policy checks.
2. Constrain the normal LM head to generate exactly one valid option index token, then map it back
   to the key. No arbitrary text token is allowed.

Start with explicit `none_of_above` training. Compute a rejection score from the margin between its
logit and the best real-option logit. Fit the threshold on validation data only. Bind the resulting
policy to the model checkpoint, prompt template, tokenizer revision, maximum length and option
catalog policy. A threshold calibrated for one setup must not be silently reused for another.

An independent answerability head is a later ablation, not part of the first implementation.

## Training data

Each example contains:

```text
state, question, [{key, description}, ...], target_key
```

The training mixture should include:

- ordinary in-domain choices;
- semantically close hard negatives;
- different option counts and randomized option order;
- paraphrased descriptions for the same behavior;
- unseen keys whose descriptions remain meaningful;
- far-OOD and near-domain ID-OOS examples targeting `none_of_above`;
- examples where an attractive keyword appears in the wrong option;
- counterfactual examples that change only one decisive phrase.

Use grouped cross-entropy over the runtime options. Add the permutation consistency term only after
the basic objective is stable. Keep classification temperature fitting separate from rejection
threshold fitting.

## RTX 4060 experiment

Begin with `Qwen2.5-0.5B` or a similarly sized permissively licensed decoder model:

- 4-bit QLoRA base weights;
- LoRA on attention and MLP projections;
- train the dynamic option projection and logit scale in BF16/FP16;
- sequence length 512 initially;
- micro-batch 1–2 with gradient accumulation;
- gradient checkpointing;
- 8 to 16 options per training group before testing larger catalogs.

If 0.5B underfits, try the 1.5B model with shorter sequences and the same QLoRA setup. Do not move
to a larger model until the 0.5B run establishes the data and evaluation pipeline.

## Evaluation contract

Report all of the following on untouched tests:

- known-option accuracy;
- accuracy on option descriptions never seen during training;
- option-order consistency;
- far-OOD recall and false-reject rate;
- hard ID-OOS recall and false-reject rate;
- NLL, multiclass Brier score and ECE;
- risk/coverage at a fixed error budget;
- latency and peak VRAM by option count and prompt length.

The minimum comparisons are the current MiniLM cross-encoder, cached bi-encoder and this LLM head.
Use the same train, validation and test identities. A useful result must improve hard ID-OOS and
unseen-option performance, not only the easy far-OOD score.

## Expected trade-off

Compared with the cached bi-encoder, this architecture gives up option-vector caching and some
latency. In return, the LLM processes the query, instructions and option wording jointly, which is
the capability needed for new option descriptions and subtle distinctions. For very large catalogs,
the bi-encoder can remain a retrieval stage and the LLM can score only the retrieved shortlist, but
shortlist recall must be reported as part of the end-to-end benchmark.
