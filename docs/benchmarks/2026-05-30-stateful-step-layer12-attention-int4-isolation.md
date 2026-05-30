# Stateful Step Layer12 Attention Int4 Isolation

Date: 2026-05-30

## Scope

This note records a middle-layer attention int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It follows the layer12 FFN-only experiment. Layer12 FFN-only int4 failed, so
this experiment tests whether attention projections are a safer int4 entry
point than FFN projections.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer12-attention-int4.json
```

Policy id:

```text
stateful-step-layer12-attention-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-11:     attention + FFN fp16
layer 12:        attention Q/K/O + V int4, FFN fp16
layers 13-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 4
selected layer:   12
selected components:
  attentionQKO: 3
  attentionV:   1
selected op names:
  model_model_layers_12_self_attn_q_proj_weight
  model_model_layers_12_self_attn_k_proj_weight
  model_model_layers_12_self_attn_v_proj_weight
  model_model_layers_12_self_attn_o_proj_weight_promoted_to_fp16
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer12-attention-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer12-attention-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer12-attention-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,151,555,238
compiled artifact:           2.0 GB
artifact total in benchmark: 2,161,479,310 bytes
```

Compressing one attention layer is a sensitivity probe only. It does not
materially reduce the deployable artifact size.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer12-attention-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  candidate [5, 24, 49, 11127, 45050]
  overlap:  5/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  candidate [285, 316, 1070, 976, 3212]
  overlap:  5/5

prefix 4:
  fp16      [9622, 14504, 448, 1690, 15046]
  candidate [9622, 14504, 448, 15046, 1690]
  overlap:  5/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [3732, 1674, 1494, 242, 3724]
  overlap:  4/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [36734, 2319, 2218, 32430, 2242]
  overlap:  4/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [280, 285, 691, 450, 7287]
  overlap:  5/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [1974, 591, 343, 416, 2452]
  overlap:  5/5
```

Unlike layer12 FFN-only int4, layer12 attention-only int4 preserves the final
prompt top-5 exactly and keeps high overlap across intermediate prefixes.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer12-attention-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [1974, 10300]
teacher IDs:   [1974, 10300]
token agreement: 1.0
first token: 246.75 ms
decode throughput: 55.07 tok/s
peak RSS: 2157.89 MB
```

## Interpretation

This is the first middle-layer int4 isolation result that preserves the teacher
tokens on `en-short-001`. It gives a concrete direction for the next search:
middle-layer attention projections are safer int4 candidates than FFN
projections under the current Core ML post-conversion palettization path.

This does not promote a deployable policy yet. The artifact is still about 2 GB
because only one attention layer was compressed, and the result has only been
checked on one prompt. It should advance to a wider attention window only after
nearby attention layers show similar prefix stability.

Next candidates:

```text
layer 10-13 attention-only int4
layer 8-15 attention-only int4
attention Q/K/O-only vs V-only if the wider window starts drifting
```
