# Stateful Step Layer10-13 V-Only Attention Int4

Date: 2026-05-30

## Scope

This note records the next V-only expansion after the layer11-12 projection
attribution experiment.

Layer11-12 Q/K/O-only int4 failed, while layer11-12 V-only int4 passed. This
experiment tests whether the safe V-only axis can expand from two middle layers
to a four-layer middle window without repeating the full attention-window
collapse.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer10-13-attention-v-int4.json
```

Policy id:

```text
stateful-step-layer10-13-attention-v-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-9:      attention + FFN fp16
layers 10-13:    attention V int4, attention Q/K/O fp16, FFN fp16
layers 14-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 4
selected component: attentionV
selected layers:    10, 11, 12, 13
selected op names:
  model_model_layers_10_self_attn_v_proj_weight
  model_model_layers_11_self_attn_v_proj_weight
  model_model_layers_12_self_attn_v_proj_weight
  model_model_layers_13_self_attn_v_proj_weight
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer10-13-attention-v-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer10-13-attention-v-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer10-13-attention-v-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,159,812,818
compiled artifact:           2.0 GB
artifact total in benchmark: 2,169,736,888 bytes
```

This is still a sensitivity probe. Four V projections are too small a fraction
of the model to materially change deployable size.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer10-13-attention-v-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  candidate [5, 24, 49, 11127, 45050]
  overlap:  5/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  candidate [285, 1070, 316, 3212, 976]
  overlap:  5/5

prefix 4:
  fp16      [9622, 14504, 448, 1690, 15046]
  candidate [9622, 14504, 448, 15046, 1690]
  overlap:  5/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [3732, 1674, 1494, 242, 2790]
  overlap:  5/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [36734, 2319, 2242, 2218, 3229]
  overlap:  5/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [280, 285, 691, 450, 7287]
  overlap:  5/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [1974, 591, 343, 416, 2452]
  overlap:  5/5
```

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer10-13-attention-v-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [1974, 10300]
teacher IDs:   [1974, 10300]
token agreement: 1.0
first token: 266.17 ms
decode throughput: 86.17 tok/s
peak RSS: 2168.84 MB
```

## Interpretation

The V-only axis remains stable when widened from layers 11-12 to layers 10-13.
This strengthens the projection-attribution conclusion:

```text
V projection int4 is currently the safest attention subcomponent.
Q/K/O should stay protected until Q/K versus O attribution or calibrated
activation-aware quantization is available.
```

Next useful candidate:

```text
layer8-15 V-only int4
```

If layer8-15 V-only also passes, V-only can become the first attention
subcomponent to test on a category-balanced prompt batch. If it fails, the next
step should be an activation-aware sensitivity scorer rather than more window
widening.
