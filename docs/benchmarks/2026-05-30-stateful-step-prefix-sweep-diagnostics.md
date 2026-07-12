# Stateful Step Prefix Sweep Diagnostics

Date: 2026-05-30

## Scope

This note records the first Swift-side prompt-prefix sweep for the
`stateful-step-kv` Core ML route.

It is separate from the single full-prompt logits diagnostic. The goal here is to
find how early global int4 begins to diverge from the fp16 stateful-step
reference.

## Diagnostic Mode

`WatchLMBenchmark` now accepts:

```text
--diagnostics-prefix-lengths 1,2,4,8,12,16,18
```

When combined with `--diagnostics-top-k`, the Swift runner tokenizes each prompt
once, slices token ID prefixes, and runs diagnostics on each prefix.

This keeps the tokenizer, Core ML graph interface selection, stateful-step
runtime contract, and top-k extraction inside the Swift inference stack.

## Artifacts

FP16:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256/stateful-step-kv-256.mlmodelc
artifacts/benchmarks/stateful-step-kv-256-fp16-prefix-logits-diagnostics.json
```

Global int4:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlmodelc
artifacts/benchmarks/stateful-step-kv-256-int4-prefix-logits-diagnostics.json
```

Prompt:

```text
en-short-001
token count: 18
prefix lengths: 1, 2, 4, 8, 12, 16, 18
```

## Results

### FP16

```text
prefix 1:  prefill top-5 [5, 24, 49, 11127, 45050], margin 0.3125
prefix 2:  prefill top-5 [285, 1070, 316, 3212, 976], margin 1.2500
prefix 4:  prefill top-5 [9622, 14504, 448, 1690, 15046], margin 2.7656
prefix 8:  prefill top-5 [3732, 1674, 242, 1494, 2790], margin 0.6680
prefix 12: prefill top-5 [36734, 2319, 2242, 3229, 2218], margin 0.3594
prefix 16: prefill top-5 [280, 285, 691, 450, 7287], margin 1.0703
prefix 18: prefill top-5 [1974, 591, 343, 416, 2452], margin 0.4609
```

### Global Int4

```text
prefix 1:  prefill top-5 [5, 24, 121400, 13626, 8], margin 12.0000
prefix 2:  prefill top-5 [34, 68268, 29683, 9559, 64584], margin 4.9141
prefix 4:  prefill top-5 [130073, 52, 678, 316, 442], margin 3.4219
prefix 8:  prefill top-5 [262, 112368, 92109, 6304, 53814], margin 0.4219
prefix 12: prefill top-5 [12316, 76065, 115463, 95236, 53624], margin 0.7344
prefix 16: prefill top-5 [28273, 77, 30079, 24103, 50341], margin 1.4219
prefix 18: prefill top-5 [5, 121400, 24, 26966, 8], margin 11.2422
```

## Interpretation

The fp16 and global-int4 artifacts agree on the prefill top-1 only for the
single-token prefix. At prefix 2, the int4 graph already picks a different
prefill token.

This narrows the quantization problem:

- drift is visible before long-context accumulation
- drift is visible before the full 18-token prompt
- global int4 produces overconfident wrong logits, for example prefix 18 has an
  int4 top-1 margin of 11.2422 versus fp16 margin 0.4609

The next quantization pass should avoid treating the full model as uniformly
compressible. The first follow-up should test protected embeddings and early
layers before spending time on later-layer-only policies.
