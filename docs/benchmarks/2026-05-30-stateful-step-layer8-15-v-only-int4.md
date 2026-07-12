# Stateful Step Layer8-15 V-Only Attention Int4

Date: 2026-05-30

## Scope

This note records a wider V-only attention int4 experiment for the shared
`stateful-step-kv` Core ML route.

Layer11-12 V-only passed, then layer10-13 V-only passed. This experiment tests
whether the same safe axis can expand to layers 8-15 and whether it preserves a
category-balanced prompt batch compared with fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer8-15-attention-v-int4.json
```

Policy id:

```text
stateful-step-layer8-15-attention-v-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-7:      attention + FFN fp16
layers 8-15:     attention V int4, attention Q/K/O fp16, FFN fp16
layers 16-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 8
selected component: attentionV
selected layers:    8, 9, 10, 11, 12, 13, 14, 15
selected op names:
  model_model_layers_8_self_attn_v_proj_weight
  model_model_layers_9_self_attn_v_proj_weight
  model_model_layers_10_self_attn_v_proj_weight
  model_model_layers_11_self_attn_v_proj_weight
  model_model_layers_12_self_attn_v_proj_weight
  model_model_layers_13_self_attn_v_proj_weight
  model_model_layers_14_self_attn_v_proj_weight
  model_model_layers_15_self_attn_v_proj_weight
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer8-15-attention-v-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer8-15-attention-v-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer8-15-attention-v-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,157,454,606
compiled artifact:           2.0 GB
artifact total in benchmark: 2,167,378,806 bytes
```

This is still not a Watch SE deployable artifact. It is a direction-finding
policy that compresses only eight V projection tensors.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer8-15-attention-v-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  candidate [5, 24, 49, 11127, 45050]
  overlap:  5/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  candidate [285, 316, 1070, 3212, 976]
  overlap:  5/5

prefix 4:
  fp16      [9622, 14504, 448, 1690, 15046]
  candidate [9622, 14504, 448, 1690, 15046]
  overlap:  5/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [3732, 1674, 242, 2790, 1494]
  overlap:  5/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [36734, 2242, 2319, 2218, 3229]
  overlap:  5/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [280, 285, 691, 7287, 450]
  overlap:  5/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [1974, 591, 343, 416, 359]
  overlap:  4/5
```

The full prompt has one top-5 membership change, so this candidate is stable
enough for batch comparison but not drift-free.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer8-15-attention-v-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [1974, 10300]
teacher IDs:   [1974, 10300]
token agreement: 1.0
first token: 247.91 ms
decode throughput: 100.85 tok/s
peak RSS: 2163.89 MB
```

## Batch Gate

Candidate report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer8-15-attention-v-int4-batch10-cap2.json
```

FP16 baseline report:

```text
artifacts/benchmarks/stateful-step-kv-256-fp16-batch10-cap2.json
```

Result:

```text
candidate average token agreement: 0.9
fp16 average token agreement:      0.9
candidate succeeded prompts:       10/10
fp16 succeeded prompts:            10/10
candidate peak RSS:                2180.69 MB
fp16 peak RSS:                     2450.44 MB
candidate first token avg:         216.89 ms
fp16 first token avg:              216.32 ms
```

Per-prompt agreement matched fp16 exactly. The only 0.0 prompt was
`watch-utility-002`, which also produced 0.0 under fp16 because the runtime
stopped without emitting the teacher EOS token.

## Interpretation

Layer8-15 V-only int4 is the strongest attention-subcomponent result so far:

```text
single-prompt teacher smoke: passes
prefix diagnostics:          high overlap, one full-prompt top-5 drift
batch10 cap2:                matches fp16 agreement profile exactly
memory:                      lower than fp16 host batch RSS, but still multi-GB
```

It should not be promoted as a Watch SE artifact because the package and host
RSS remain far above device constraints. It should be promoted as a measured
component-level direction:

```text
Use V-only as the first attention class to combine with other safe compression
axes. Keep Q/K/O, FFN, lm_head, embeddings, norms, and KV state protected until
calibrated evidence says otherwise.
```
