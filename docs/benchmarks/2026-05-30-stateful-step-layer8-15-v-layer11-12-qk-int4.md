# Stateful Step Layer8-15 V Plus Layer11-12 QK Int4

Date: 2026-05-30

## Scope

This note records the first composition attempt after two individually stable
attention subcomponent results:

```text
layer8-15 V-only:       matches fp16 on batch10 cap2
layer11-12 QK-only:     matches fp16 on batch10 cap2
```

The hypothesis was:

```text
Can V-only middle compression and narrow Q/K compression compose without
triggering the earlier Q/K/O interaction failure?
```

## Policy

```text
tools/conversion/mixed-precision-policy-stateful-step-layer8-15-v-layer11-12-qk-int4.json
policy: stateful-step-layer8-15-v-layer11-12-qk-int4-rest-fp16
```

Selected weights:

```text
attentionV:   layers 8...15
attentionQKO: layers 11...12, narrowed by opNamePatterns to q_proj and k_proj
```

Protected weights:

```text
embedding, lm_head, norms, all FFN, all O projections, Q/K outside layers 11...12
```

## Selector Gate

The policy-contract test verified exact selector behavior:

```text
node --test --test-name-pattern "V plus layer11-12 QK" test/realConversionCli.test.js
```

Result:

```text
1 test passed
```

Compression audit:

```text
selectedOpCount: 12
selectedByComponent:
  attentionV:   8
  attentionQKO: 4
selectedByLayer:
  8: 1
  9: 1
  10: 1
  11: 3
  12: 3
  13: 1
  14: 1
  15: 1
```

## Core ML Artifacts

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer8-15-v-layer11-12-qk-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer8-15-v-layer11-12-qk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos-stateful-step-kv-256-layer8-15-v-layer11-12-qk-int4/stateful-step-kv-256-mixed.mlmodelc
mlpackageBytes: 2,146,838,858
```

Both macOS and watchOS compiles succeeded. `coremlc` again warned that this
stateful-step graph requires watchOS 11.0 or newer despite the requested
deployment target being 10.0.

## Prefix Gate

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer8-15-v-layer11-12-qk-int4-prefix-logits-diagnostics.json
```

Top-5 overlap against fp16:

```text
prefix   overlap
1        5/5
2        0/5
4        0/5
8        0/5
12       0/5
16       0/5
18       0/5
```

The candidate fails at the same early prefix depth as the previous unstable
attention-window candidates.

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer8-15-v-layer11-12-qk-int4-teacher-smoke.json
```

Result:

```text
generated IDs: [127412, 220]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 12404.54 ms
decode throughput: 1.05 tok/s
peak RSS: 2159.45 MB
```

Because the teacher smoke failed, the batch10 gate was intentionally skipped.

## Interpretation

This is a useful negative result:

```text
V-only can be safe.
QK-only can be safe.
V + QK in the same layer window is not safe under current Core ML int4
palettization.
```

The likely issue is interaction/accumulation, not either subcomponent alone.
In layers 11 and 12, this policy quantizes Q, K, and V together while keeping O
fp16. That is enough to collapse prefix logits at prefix 2.

## Next Consequence

Do not keep composing locally safe axes by addition. The next broad step should
be an activation-aware sensitivity scorer inspired by AWQ/imatrix-style
calibration.

If one more local attention experiment is needed, it should answer the narrower
interaction question:

```text
Does QK remain stable when V is kept fp16 specifically in layers 11 and 12?
```

That would distinguish same-layer QK/V interaction from wider V-window
interaction. It should not replace the broader calibrated sensitivity path.
