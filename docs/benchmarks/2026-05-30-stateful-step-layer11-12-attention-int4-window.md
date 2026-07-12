# Stateful Step Layer11-12 Attention Int4 Window

Date: 2026-05-30

## Scope

This note records a narrowed middle-layer attention int4 experiment for the
`stateful-step-kv` Core ML route.

Layer12 attention-only int4 preserved the teacher output. The wider layer10-13
attention window failed at prefix 2. This experiment tests whether a smaller
left-side expansion, layer11-12, can preserve logits while compressing more than
one attention layer.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-int4.json
```

Policy id:

```text
stateful-step-layer11-12-attention-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-10:     attention + FFN fp16
layers 11-12:    attention Q/K/O + V int4, FFN fp16
layers 13-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 8
selected layers:   11, 12
selected components:
  attentionQKO: 6
  attentionV:   2
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-12-attention-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer11-12-attention-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-12-attention-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,140,939,490
compiled artifact:           2.0 GB
artifact total in benchmark: 2,150,863,689 bytes
```

This is a narrowed sensitivity experiment. It is not yet a deployable SE2
candidate.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  candidate [5, 24, 49, 11127, 45050]
  overlap:  5/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  candidate [5, 24, 5298, 1207, 20773]
  overlap:  0/5

prefix 4:
  fp16      [9622, 14504, 448, 1690, 15046]
  candidate [242, 40, 4587, 1688, 124525]
  overlap:  0/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [3422, 16189, 29365, 39272, 62426]
  overlap:  0/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [6531, 75152, 3794, 14379, 3740]
  overlap:  0/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [113655, 23047, 67782, 63855, 118296]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [5, 49, 24, 1307, 45050]
  overlap:  0/5
```

The layer11-12 window fails the prefix gate at prefix 2. It behaves much closer
to the failed layer10-13 window than to the stable layer12-only experiment.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 24047]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 11329.65 ms
decode throughput: 1.04 tok/s
peak RSS: 2157.50 MB
```

## Interpretation

Layer12 attention-only int4 is stable, but adding layer11 breaks the prompt
trajectory. This narrows the likely causes:

```text
1. layer11 attention is individually sensitive under current Core ML int4
2. a two-layer attention window already accumulates too much error
```

The next experiment should distinguish those two cases before trying another
wide window:

```text
layer11 attention-only int4
layer12-13 attention-only int4
```

If layer11 alone fails, avoid the left side of layer12 and test the right side.
If layer11 alone passes but layer11-12 fails, then multi-layer attention
accumulation is the main issue and Q/K/O-only vs V-only split becomes the next
useful axis.
