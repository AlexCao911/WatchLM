# Stateful Step Layer0 FFN Int4 Isolation

Date: 2026-05-30

## Scope

This note records a component-level int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It follows the layer0 whole-layer and layer0 attention-only experiments. This
policy compresses only layer 0 FFN projections and keeps attention fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer0-ffn-int4.json
```

Policy id:

```text
stateful-step-layer0-ffn-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layer 0:         attention fp16, FFN int4
layers 1-23:     attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 3
selected layer:   0
selected components:
  FFN: 3
selected op names:
  model_model_layers_0_mlp_gate_proj_weight
  model_model_layers_0_mlp_up_proj_weight
  model_model_layers_0_mlp_down_proj_weight
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer0-ffn-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer0-ffn-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer0-ffn-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,130,321,303
compiled artifact:           2.0 GB
artifact total in benchmark: 2,140,245,332 bytes
```

This remains a sensitivity experiment. Compressing only the three layer0 FFN
weights does not materially change the deployment size.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer0-ffn-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  candidate [5, 24, 49, 45050, 11127]
  overlap:  5/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  candidate [5, 24, 5298, 20773, 1307]
  overlap:  0/5

prefix 4:
  fp16      [9622, 14504, 448, 1690, 15046]
  candidate [34621, 4227, 1707, 107680, 5839]
  overlap:  0/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [13251, 41716, 12690, 108254, 7169]
  overlap:  0/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [793, 1628, 9870, 5910, 29795]
  overlap:  0/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [856, 5472, 1979, 24727, 80739]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [5, 49, 24, 1307, 11127]
  overlap:  0/5
```

Layer0 FFN-only int4 survives the BOS-only prefix, but the trajectory diverges
immediately after prefix 2 and never recovers on this prompt.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer0-ffn-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 34764]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 9290.90 ms
decode throughput: 67.48 tok/s
peak RSS: 2142.73 MB
```

## Interpretation

Layer0 FFN-only int4 is more destructive than the layer0 attention-only policy
on this prompt. Attention-only preserved the first generated token, while
FFN-only flips the first token to `5` and has zero top-5 overlap at the full
prompt prefix.

This strengthens the edge-layer rule for the current Core ML palettization path:

```text
Do not int4-compress layer 0 FFN in SE2 candidates unless a later calibration or
different quantization method proves it can preserve prefix logits.
```

The next useful search direction is not broader layer0 compression. It is finer
non-edge isolation:

```text
middle-layer FFN-only int4
middle-layer attention-only int4
late-layer component isolation around layer 23
quality-preserving calibration before promoting any global int4 policy
```
