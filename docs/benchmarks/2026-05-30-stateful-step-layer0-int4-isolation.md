# Stateful Step Layer0 Int4 Isolation

Date: 2026-05-30

## Scope

This note records a single-layer int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It is separate from the layer23 isolation note. Layer23 tested an
output-adjacent layer. This experiment tests the opposite edge: only layer 0
attention and FFN weights are palettized to int4 while the rest of the model
remains fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer0-int4.json
```

Policy id:

```text
stateful-step-layer0-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layer 0:         attention + FFN int4
layers 1-23:     attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 7
selected layer:   0
selected components:
  attentionQKO: 3
  attentionV:   1
  FFN:          3
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer0-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer0-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer0-int4/conversion-report.json
```

Size:

```text
layer0-int4 compiled: 2.0 GB
fp16 compiled:        2.0 GB
global int4 compiled: 516 MB
```

Only one layer is compressed, so this is a sensitivity experiment, not a Watch
SE2 size candidate.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer0-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16    [5, 24, 49, 11127, 45050]
  layer0  [5, 24, 49, 45050, 11127]
  overlap: 5/5

prefix 2:
  fp16    [285, 1070, 316, 3212, 976]
  layer0  [24, 49, 5, 1307, 11127]
  overlap: 0/5

prefix 18:
  fp16    [1974, 591, 343, 416, 2452]
  layer0  [5, 24, 3492, 5298, 1176]
  overlap: 0/5
```

The layer0-only artifact survives the BOS-only prefix, but diverges immediately
after the second token.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer0-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 20790]
token agreement: 0.0
first token: 9303.40 ms
decode throughput: 1.03 tok/s
peak RSS: 2135.91 MB
artifact total: 2,129,629,713 bytes
```

## Interpretation

Layer 0 is also highly sensitive in the current Core ML int4 palettization path.
Compressing only seven layer-0 attention/FFN weights is enough to corrupt the
prompt-state trajectory by prefix 2.

Together with the layer23 isolation result, this means the stateful-step route
cannot assume "edge layer protection" alone is enough. Both the first and final
transformer layers are unsafe under current Core ML int4 compression.

The next quantization work should avoid whole-layer int4 at the transformer
edges and move to smaller component isolation, for example:

```text
layer 0 attention-only int4
layer 0 FFN-only int4
middle-layer attention-only int4
middle-layer FFN-only int4
```

Only policies that survive prefix diagnostics should advance to teacher smoke or
SE2 promotion.
