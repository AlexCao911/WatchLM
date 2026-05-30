# Stateful Step Layer0 Attention Int4 Isolation

Date: 2026-05-30

## Scope

This note records a component-level int4 isolation experiment for the
`stateful-step-kv` Core ML route.

It follows the layer0 whole-layer experiment. Instead of compressing layer 0
attention and FFN together, this policy compresses only layer 0 attention
projections and keeps FFN fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer0-attention-int4.json
```

Policy id:

```text
stateful-step-layer0-attention-int4-rest-fp16
```

Summary:

```text
embedding:       fp16
lm_head:         fp16
norms:           fp16
layer 0:         attentionQKO + attentionV int4, FFN fp16
layers 1-23:     attention + FFN fp16
KV state:        fp16
```

Compression audit:

```text
int4 selected ops: 4
selected layer:   0
selected components:
  attentionQKO: 3
  attentionV:   1
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer0-attention-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer0-attention-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer0-attention-int4/conversion-report.json
```

Size:

```text
layer0-attention-int4 compiled: 2.0 GB
artifact total in benchmark:      2,161,479,305 bytes
```

This is still a sensitivity experiment. Compressing only four weights does not
materially change the deployment size.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer0-attention-int4-prefix-logits-diagnostics.json
```

Compared with fp16:

```text
prefix 1:
  fp16      [5, 24, 49, 11127, 45050]
  candidate [24, 11127, 49, 45050, 5]
  overlap:  5/5

prefix 2:
  fp16      [285, 1070, 316, 3212, 976]
  candidate [11127, 24, 1307, 49, 45050]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [1974, 591, 416, 343, 359]
  overlap:  4/5
```

The intermediate prefixes are mixed: prefix 2 diverges completely, while the
full prompt recovers the fp16 top-1 token.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer0-attention-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [1974, 220]
teacher IDs:   [1974, 10300]
token agreement: 0.5
first token: 235.36 ms
decode throughput: 99.91 tok/s
peak RSS: 2157.08 MB
```

## Interpretation

Layer0 attention-only int4 is less destructive than compressing the whole first
layer. It preserves the full-prompt first token, unlike layer0 whole-layer int4.

It is still not quality-safe. The second generated token diverges from the
teacher, and prefix 2 has zero top-5 overlap with fp16.

This narrows the next search direction:

```text
1. Test layer0 FFN-only int4 to separate FFN contribution from attention.
2. Test attention subgroups if needed: q/k/o only vs v only.
3. Avoid promoting attention-only policies from a single first-token match.
```

The useful signal is not "layer0 attention works"; it is "layer0 attention is
less catastrophic than whole-layer int4 and deserves finer subgroup testing."
