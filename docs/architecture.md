# Architecture notes

## Version 0: cross-encoder baseline

The baseline scores every candidate independently:

```text
state + question + candidate option
                |
             MiniLM
                |
           scalar logit
```

For a mutually exclusive Choice, logits from all caller-declared candidates are normalized with a
softmax. The service returns the winning caller-provided key and the full distribution. There is no
text decoder.

This baseline intentionally recomputes the state for every candidate. It is inefficient but gives
the cleanest accuracy baseline and keeps the initial experiment small.

## Version 1: cached queries and shared state

If the baseline succeeds, split it into:

```text
state -> state encoder -> state tokens -------------------+
                                                          |
question + option -> query encoder -> decision token -> cross attention -> logit
```

Static question and option representations can be cached, while cross-attention against the live
state remains dynamic. All decision tokens can attend to the same state in one batch. A learned
`[DECIDE]` token is preferable to treating `[SEP]` as a pooling token, because `[SEP]` is only a
delimiter unless training explicitly gives it the decision role.

For large tool catalogs, use a dual encoder to retrieve a shortlist and the cross-encoder to rerank
it. Measure the recall lost by retrieval before claiming an end-to-end speedup.

## Output heads

- **Choice**: softmax across mutually exclusive options.
- **Noul**: one sigmoid for an independent yes/no statement.
- **Score**: ordinal thresholds or a softmax over ordered levels followed by an expectation.
- **Abstain**: an explicit `none_of_the_above` option or a separately trained answerability head.

Raw softmax values are not automatically calibrated. Fit temperature scaling on a held-out split
and report Brier score, NLL, ECE and selective risk/coverage on untouched test data.

