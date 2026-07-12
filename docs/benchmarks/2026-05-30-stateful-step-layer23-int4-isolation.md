# Stateful Step Layer23 Int4 Isolation

Date: 2026-05-30

## Scope

This note records the first single-layer int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It is separate from the early4 experiment. The early4 policy tested a broad
20-layer int4 region. This note tests the smallest useful late-layer cut: only
layer 23 attention and FFN weights are palettized to int4 while the rest of the
model remains fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer23-int4.json
```

Policy id:

```text
stateful-step-layer23-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-22:     attention + FFN fp16
layer 23:        attention + FFN int4
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 7
selected layer:   23
selected components:
  attentionQKO: 3
  attentionV:   1
  FFN:          3
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer23-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer23-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer23-int4/conversion-report.json
```

Size:

```text
layer23-int4 compiled: 2.0 GB
fp16 compiled:         2.0 GB
global int4 compiled:  516 MB
```

Only one layer is compressed, so this is a sensitivity experiment, not a size
candidate for Watch SE2.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer23-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  layer23   [5, 24, 49, 2331, 608]
  overlap:  3/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  layer23   [5, 49, 20773, 24, 31]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  layer23   [5, 24, 49, 2331, 608]
  overlap:  0/5
```

The layer23-only int4 artifact diverges by prefix 2, just like the broader
global-int4 and early4-int4 experiments.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer23-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 5]
token agreement: 0.0
first token: 9304.85 ms
decode throughput: 1.05 tok/s
peak RSS: 2132.88 MB
artifact total: 2,129,629,726 bytes
```

## Interpretation

Layer 23 is highly sensitive in the current Core ML int4 palettization path.
Compressing only seven layer-23 attention/FFN weights is enough to flip the first
generated token away from the fp16 baseline.

This rules out any near-term policy that includes the final transformer layer in
int4. Future isolation should test earlier single layers or small earlier layer
groups, while keeping the final layers and lm_head fp16.

This also explains why broad int4 policies fail so sharply: they include the
output-adjacent layer, and that alone can corrupt the final logits.
