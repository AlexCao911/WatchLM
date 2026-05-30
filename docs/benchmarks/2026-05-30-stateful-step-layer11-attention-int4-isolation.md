# Stateful Step Layer11 Attention Int4 Isolation

Date: 2026-05-30

## Scope

This note records a layer11 attention-only int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It follows the failed layer11-12 attention window. Layer12 attention-only int4
had already passed, while layer11-12 failed at prefix 2. This experiment checks
whether layer11 itself is unsafe, or whether the failure is more likely caused
by adjacent-layer accumulation.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-attention-int4.json
```

Policy id:

```text
stateful-step-layer11-attention-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-10:     attention + FFN fp16
layer 11:        attention Q/K/O + V int4, FFN fp16
layers 12-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 4
selected layer:   11
selected components:
  attentionQKO: 3
  attentionV:   1
selected op names:
  model_model_layers_11_self_attn_q_proj_weight
  model_model_layers_11_self_attn_k_proj_weight
  model_model_layers_11_self_attn_v_proj_weight
  model_model_layers_11_self_attn_o_proj_weight_promoted_to_fp16
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-attention-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer11-attention-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-attention-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,151,555,238
compiled artifact:           2.0 GB
artifact total in benchmark: 2,161,479,305 bytes
```

As with layer12 attention-only, this is a sensitivity probe. Compressing one
attention layer does not materially reduce the deployable artifact size.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-attention-int4-prefix-logits-diagnostics.json
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
  candidate [9622, 14504, 448, 1690, 15046]
  overlap:  5/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [3732, 1674, 2790, 242, 1494]
  overlap:  5/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [36734, 2319, 3229, 2242, 2218]
  overlap:  5/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [280, 285, 691, 7287, 450]
  overlap:  5/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [1974, 591, 343, 416, 2452]
  overlap:  5/5
```

Layer11 attention-only preserves fp16 top-5 membership at every tested prefix.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-attention-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [1974, 10300]
teacher IDs:   [1974, 10300]
token agreement: 1.0
first token: 282.20 ms
decode throughput: 81.01 tok/s
peak RSS: 2154.88 MB
```

## Interpretation

Layer11 attention-only is stable on this probe. Since layer12 attention-only
also passed, but layer11-12 attention-only failed at prefix 2, the failure is
more likely caused by adjacent-layer error accumulation or by a specific
Q/K/O/V interaction than by layer11 or layer12 being individually unsafe.

This result makes more blind window widening less attractive. The next useful
step should be driven by stronger priors: calibrated/groupwise quantization
evidence, per-tensor sensitivity scoring, or a structurally motivated split
between Q/K/O and V.
