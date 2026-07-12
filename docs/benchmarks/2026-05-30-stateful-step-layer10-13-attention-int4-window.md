# Stateful Step Layer10-13 Attention Int4 Window

Date: 2026-05-30

## Scope

This note records the first widened middle-layer attention int4 experiment for
the `stateful-step-kv` Core ML route.

Layer12 attention-only int4 preserved the teacher output on `en-short-001`.
This experiment widens that stable axis to layers 10-13 while keeping all FFN
weights, embeddings, lm head, norms, and KV state at fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer10-13-attention-int4.json
```

Policy id:

```text
stateful-step-layer10-13-attention-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layers 0-9:      attention + FFN fp16
layers 10-13:    attention Q/K/O + V int4, FFN fp16
layers 14-23:    attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 16
selected layers:   10, 11, 12, 13
selected components:
  attentionQKO: 12
  attentionV:    4
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer10-13-attention-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer10-13-attention-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer10-13-attention-int4/conversion-report.json
```

Size:

```text
mlpackage bytes:             2,119,707,994
compiled artifact:           2.0 GB
artifact total in benchmark: 2,129,632,448 bytes
```

The window begins to reduce size versus the fp16 stateful-step artifact, but it
is still a sensitivity experiment rather than a deployable SE2 candidate.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer10-13-attention-int4-prefix-logits-diagnostics.json
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
  candidate [947, 285, 1974, 2178, 678]
  overlap:  0/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [242, 220, 282, 29365, 17854]
  overlap:  1/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [12935, 3237, 45532, 10295, 112547]
  overlap:  0/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [5, 24, 3492, 1176, 5298]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [5, 24, 49, 1307, 45050]
  overlap:  0/5
```

The widened window fails the prefix gate at prefix 2. This means it should not
be widened further to layer8-15.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer10-13-attention-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 25416]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 10291.90 ms
decode throughput: 43.93 tok/s
peak RSS: 2136.80 MB
```

## Interpretation

Layer12 attention-only int4 was stable, but the layer10-13 attention window is
not. This narrows the search rather than ending it:

```text
Do not expand directly from one stable middle attention layer to a four-layer
attention window.
```

The failure could come from either:

```text
1. a sensitive neighboring attention layer, likely 10, 11, or 13
2. accumulated error from multiple attention layers being palettized together
3. one attention subgroup such as V or Q/K/O being less stable in the wider window
```

The next useful candidates should therefore shrink or split the window:

```text
layer11-12 attention-only int4
layer12-13 attention-only int4
layer10, layer11, and layer13 single-layer attention-only int4 if needed
Q/K/O-only vs V-only after identifying the unstable side of the window
```
