# Stateful Step Layer12 FFN Int4 Isolation

Date: 2026-05-30

## Scope

This note records a middle-layer FFN int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It follows the layer0 FFN-only experiment. The goal is to separate "edge layer
sensitivity" from "FFN int4 palettization sensitivity" by compressing only the
three FFN projections in layer 12.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer12-ffn-int4.json
```

Policy id:

```text
stateful-step-layer12-ffn-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-11:     attention + FFN fp16
layer 12:        attention fp16, FFN int4
layers 13-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 3
selected layer:   12
selected components:
  FFN: 3
selected op names:
  model_model_layers_12_mlp_gate_proj_weight
  model_model_layers_12_mlp_up_proj_weight
  model_model_layers_12_mlp_down_proj_weight
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer12-ffn-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer12-ffn-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer12-ffn-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,130,321,303
compiled artifact:           2.0 GB
artifact total in benchmark: 2,140,245,342 bytes
```

Compressing one FFN layer is a sensitivity probe only. It does not materially
reduce the deployable artifact size.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer12-ffn-int4-prefix-logits-diagnostics.json
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
  candidate [84, 1688, 34, 242, 388]
  overlap:  0/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [242, 4540, 282, 18099, 322]
  overlap:  1/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [387, 4969, 96214, 1419, 18784]
  overlap:  0/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [10285, 2282, 2096, 9241, 2400]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [5, 24, 49, 1307, 45050]
  overlap:  0/5
```

Layer12 FFN-only int4 preserves the BOS-only prefix, but it diverges at prefix 2
and does not recover at the full prompt.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer12-ffn-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 663]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 10299.77 ms
decode throughput: 64.05 tok/s
peak RSS: 2147.38 MB
```

## Interpretation

This result weakens the hypothesis that only edge FFN layers are unsafe. A
middle FFN layer compressed to Core ML int4 is enough to corrupt the prompt
trajectory for `en-short-001`.

For the current stateful-step Core ML route, the near-term rule should be:

```text
Do not use uncalibrated Core ML int4 palettization for MiniCPM FFN projections
in SE2 candidates.
```

The next search should move in one of two directions:

```text
1. Test attention-only middle-layer int4, because layer0 attention-only was less
   destructive than FFN-only.
2. Investigate calibrated or groupwise quantization before Core ML conversion
   instead of relying on post-conversion kmeans palettization for FFN weights.
```
