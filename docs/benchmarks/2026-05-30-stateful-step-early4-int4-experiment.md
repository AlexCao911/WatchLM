# Stateful Step Early4 Int4 Experiment

Date: 2026-05-30

## Scope

This note records the first `stateful-step-kv` mixed int4 experiment after the
prefix sweep showed global int4 diverging by prefix 2.

It is separate from the prefix sweep note. The prefix sweep located the symptom.
This note tests one concrete mitigation: keep embeddings, lm_head, norms, and
the first four transformer layers at fp16, then palettize the remaining
attention and FFN weights to int4.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-early4-int4.json
```

Policy id:

```text
stateful-step-fp16-embed-lmhead-early4-rest-int4
```

Summary:

```text
embedding:    fp16
lm_head:      fp16
norms:        fp16
layers 0-3:   attention + FFN fp16
layers 4-23:  attention + FFN int4
KV state:     fp16
```

Compression audit:

```text
int4 selected ops: 140
selected layers:   4 through 23
selected per layer: 7
selected components:
  attentionQKO: 60
  attentionV:   20
  FFN:          60
```

## Artifacts

Source:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256/stateful-step-kv-256.mlpackage
```

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-early4-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-early4-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-early4-int4/conversion-report.json
```

Size:

```text
early4-int4 compiled: 1.2 GB
global int4 compiled: 516 MB
fp16 compiled:        2.0 GB
```

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-early4-int4-prefix-logits-diagnostics.json
```

Compared with the existing fp16 and global-int4 prefix diagnostics:

```text
prefix 1:
  fp16   top-5 [5, 24, 49, 11127, 45050]
  int4   top-5 [5, 24, 121400, 13626, 8]
  early4 top-5 [5, 45050, 42080, 122686, 59133]

prefix 2:
  fp16   top-5 [285, 1070, 316, 3212, 976]
  int4   top-5 [34, 68268, 29683, 9559, 64584]
  early4 top-5 [34, 29, 6676, 1596, 52]

prefix 18:
  fp16   top-5 [1974, 591, 343, 416, 2452]
  int4   top-5 [5, 121400, 24, 26966, 8]
  early4 top-5 [5, 1226, 122686, 20495, 11853]
```

Top-k overlap with fp16:

```text
prefix 1:  global int4 2/5, early4-int4 2/5
prefix 2:  global int4 0/5, early4-int4 0/5
prefix 4:  global int4 0/5, early4-int4 0/5
prefix 8:  global int4 0/5, early4-int4 1/5
prefix 12: global int4 0/5, early4-int4 0/5
prefix 16: global int4 0/5, early4-int4 0/5
prefix 18: global int4 0/5, early4-int4 0/5
```

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-early4-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 66747]
token agreement: 0.0
first token: 9362.29 ms
decode throughput: 1.05 tok/s
peak RSS: 1380.34 MB
artifact total: 1,322,790,710 bytes
```

## Interpretation

This policy is not viable for Watch SE2 promotion.

It answers a useful question, though: protecting embeddings, lm_head, norms, and
the first four layers does not recover fp16-aligned logits. The divergence still
appears at prefix 2, and the full prompt still selects token `5` instead of the
fp16 token `1974`.

The remaining int4-compressed layers 4-23 are enough to corrupt the prompt-state
trajectory, even for very short prefixes. The next quantization work should move
from broad mixed policies to isolation sweeps:

- quantize one late layer group at a time
- compare prefix top-k after each group
- only expand the int4 region after top-k agreement survives
