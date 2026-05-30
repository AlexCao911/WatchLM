# Stateful Step Layer11-12 Attention Projection Attribution

Date: 2026-05-30

## Scope

This note records the Q/K/O-only versus V-only attribution experiment for the
failed layer11-12 attention int4 window.

Layer11 attention-only and layer12 attention-only each passed, while layer11-12
attention-only failed. Grouped-channel no-scale palettization also failed. The
next structural question was whether the failure came from attention-score /
output projections, or from value projections.

## Policies

Q/K/O-only:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-qko-int4.json
```

```text
layers 11-12:
  attention Q/K/O: int4
  attention V:     fp16
  FFN:             fp16
everything else:   fp16
KV state:          fp16
```

V-only:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-v-int4.json
```

```text
layers 11-12:
  attention Q/K/O: fp16
  attention V:     int4
  FFN:             fp16
everything else:   fp16
KV state:          fp16
```

## Conversion Audit

Q/K/O-only:

```text
selected ops:        6
selected components: attentionQKO: 6
selected layers:     layer11: 3, layer12: 3
mlpackage bytes:     2,142,118,640
compiled artifact:   2.0 GB
```

V-only:

```text
selected ops:        2
selected components: attentionV: 2
selected layers:     layer11: 1, layer12: 1
mlpackage bytes:     2,160,991,924
compiled artifact:   2.0 GB
```

## Prefix Diagnostics

Q/K/O-only report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-qko-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:  overlap 5/5
prefix 2:  overlap 0/5
prefix 4:  overlap 0/5
prefix 8:  overlap 1/5
prefix 12: overlap 0/5
prefix 16: overlap 0/5
prefix 18: overlap 0/5
```

V-only report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-v-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:  overlap 5/5
prefix 2:  overlap 5/5
prefix 4:  overlap 5/5
prefix 8:  overlap 5/5
prefix 12: overlap 5/5
prefix 16: overlap 5/5
prefix 18: overlap 5/5
```

## Teacher Smoke

Q/K/O-only report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-qko-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 46431]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 11315.48 ms
decode throughput: 41.29 tok/s
peak RSS: 2146.25 MB
```

V-only report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-v-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [1974, 10300]
teacher IDs:   [1974, 10300]
token agreement: 1.0
first token: 275.64 ms
decode throughput: 91.73 tok/s
peak RSS: 2164.58 MB
```

## Interpretation

The layer11-12 adjacent attention failure is primarily attributable to Q/K/O,
not V.

V-only int4 across layers 11-12 preserves prefix top-5 membership at every
tested prefix and exactly matches the teacher smoke tokens. Q/K/O-only collapses
at prefix 2, matching the failure shape of the full layer11-12 attention window.

This gives a better next search direction:

```text
Do not treat attention as one component.
Keep Q/K/O protected until split further.
Use V-only as the first expandable attention subcomponent.
```

Next high-signal candidates:

```text
layer10-13 V-only int4
layer8-15 V-only int4 if layer10-13 passes
Q/K-only vs O-only if we need to recover Q/K/O compression later
```
