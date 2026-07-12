# Stateful Step Quantization Diagnosis

Date: 2026-05-30

## Scope

This note records the first stateful-step quantization matrix for context 256.

It is separate from the stateful-step runtime contract and from the first int4
artifact evidence. This document answers whether the stateful-step graph's
quality issue is caused by graph semantics or by weight compression.

## Baseline

The uncompressed `stateful-step-kv` graph is semantically valid against the
PyTorch teacher:

```text
artifact:
artifacts/coreml/compiled-macos-stateful-step-kv-256/stateful-step-kv-256.mlmodelc

report:
artifacts/benchmarks/stateful-step-kv-256-fp16-teacher-smoke.json

generated IDs: [1974, 10300]
token agreement: 1.0
first token: 280.82 ms
decode throughput: 99.68 tok/s
peak RSS: 2319.02 MB
artifact total: 2,172,093,745 bytes
```

This proves the `stateful-step-kv` graph contract, Swift prompt-token loop,
Core ML state update, logits sampling, tokenizer, and teacher comparison are
aligned for the smoke prompt.

## Failed Compression Policies

### Global Int4

```text
report:
artifacts/benchmarks/stateful-step-kv-256-int4-teacher-smoke.json

generated IDs: [5, 67778]
token agreement: 0.0
peak RSS: 651.25 MB
artifact total: 551,309,343 bytes
```

Global int4 is small enough to be interesting for SE2 packaging, but it is not
quality-safe.

### Protected No-Int4 Int8

Policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-protected-no-int4.json
```

This policy keeps attention, norms, and KV state at fp16, while quantizing
embedding, lm_head, and all FFN projections to int8.

```text
artifact:
artifacts/coreml/compiled-macos-stateful-step-kv-256-protected-no-int4/stateful-step-kv-256-mixed.mlmodelc

report:
artifacts/benchmarks/stateful-step-kv-256-protected-no-int4-teacher-smoke.json

compression audit:
int8 selected ops: 74
embedding ops: 1
FFN ops: 72
lm_head ops: 1
int4 selected ops: 0

generated IDs: [5, 3294]
token agreement: 0.0
first token: 43,229.04 ms
decode throughput: 0.97 tok/s
peak RSS: 2083.80 MB
artifact total: 1,262,462,587 bytes
```

This policy is larger and slower than global int4 while still failing quality.
It should not be promoted.

### FFN-Only Int8

Policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-ffn-int8.json
```

This keeps embedding, lm_head, attention, norms, and KV state at fp16. Only FFN
projection weights are int8.

```text
artifact:
artifacts/coreml/compiled-macos-stateful-step-kv-256-ffn-int8/stateful-step-kv-256-mixed.mlmodelc

report:
artifacts/benchmarks/stateful-step-kv-256-ffn-int8-teacher-smoke.json

compression audit:
int8 selected ops: 72
FFN ops: 72
int4 selected ops: 0

generated IDs: [5, 5]
token agreement: 0.0
first token: 31,139.51 ms
decode throughput: 1.04 tok/s
peak RSS: 3762.53 MB
artifact total: 1,663,020,222 bytes
```

This isolates the failure to FFN int8 compression; embedding and lm_head int8 are
not required to reproduce the top-1 flip.

### Single-Layer FFN12 Int8

Policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-ffn12-int8.json
```

Only layer 12 FFN gate/up/down projections are int8.

```text
artifact:
artifacts/coreml/compiled-macos-stateful-step-kv-256-ffn12-int8/stateful-step-kv-256-mixed.mlmodelc

report:
artifacts/benchmarks/stateful-step-kv-256-ffn12-int8-teacher-smoke.json

compression audit:
int8 selected ops: 3
selected layer: 12
int4 selected ops: 0

generated IDs: [5, 5]
token agreement: 0.0
first token: 10,310.16 ms
decode throughput: 48.36 tok/s
peak RSS: 2573.16 MB
artifact total: 2,150,883,473 bytes
```

Even one int8 FFN layer is enough to flip the first token in this stateful-step
Core ML path. It saves only about 21 MB versus fp16, so it is not an acceptable
tradeoff.

## Interpretation

The stateful-step graph itself is correct. The quality break appears when Core ML
weight compression rewrites FFN projections. The error then compounds across the
prompt-token state-building loop and changes the first sampled token.

For the current Core ML stateful-step route:

- fp16 is quality-correct but too large for Watch SE2
- global int4 is compact but not quality-safe
- protected int8 policies do not recover quality
- FFN int8 is not safe, even for a single layer in the smoke prompt

## Consequence

The next deployable path should not assume standard Core ML int8/int4 weight
compression can preserve MiniCPM5-1B behavior in the stateful-step graph.

Next work should investigate:

1. Core ML palettization or quantization settings that are less destructive than
   the current linear int8 FFN path.
2. Smaller context or prompt-prefill variants only after preserving fp16 quality.
3. A model-side low-bit runtime with explicit dequantization, if Core ML's
   compressed-weight operators remain too lossy for this graph.
4. Layer-by-layer logits diagnostics before and after FFN quantization to confirm
   whether one layer's output drift or Core ML compressed-op runtime behavior is
   the direct cause.
