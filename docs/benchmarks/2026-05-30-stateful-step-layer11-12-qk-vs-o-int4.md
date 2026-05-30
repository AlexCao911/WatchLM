# Stateful Step Layer11-12 QK vs O Int4 Attribution

Date: 2026-05-30

## Scope

This note records the follow-up attribution after layer11-12 Q/K/O-only int4
failed while layer11-12 V-only int4 passed.

The question was:

```text
Inside the failed Q/K/O group, is attention-score formation or output mixing
the more promising compression axis?
```

## Policies

QK-only:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-qk-int4.json
policy: stateful-step-layer11-12-attention-qk-int4-rest-fp16
selected ops: layer11/12 q_proj and k_proj
```

O-only:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-o-int4.json
policy: stateful-step-layer11-12-attention-o-int4-rest-fp16
selected ops: layer11/12 o_proj
```

Everything else stays fp16.

## Selector Gate

The policy-contract tests verified exact selector behavior:

```text
node --test --test-name-pattern "QK-only|O-only" test/realConversionCli.test.js
```

Result:

```text
3 tests passed
```

Compression audits:

```text
QK-only selectedOpCount: 4
QK-only selectedByLayer: { 11: 2, 12: 2 }

O-only selectedOpCount: 2
O-only selectedByLayer: { 11: 1, 12: 1 }
```

## Core ML Artifacts

QK-only:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-12-attention-qk-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer11-12-attention-qk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos-stateful-step-kv-256-layer11-12-attention-qk-int4/stateful-step-kv-256-mixed.mlmodelc
mlpackageBytes: 2,151,555,282
```

O-only:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-12-attention-o-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer11-12-attention-o-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos-stateful-step-kv-256-layer11-12-attention-o-int4/stateful-step-kv-256-mixed.mlmodelc
mlpackageBytes: 2,152,734,388
```

Both macOS and watchOS compiles succeeded. `coremlc` warned that this
stateful-step graph requires watchOS 11.0 or newer despite the requested
deployment target being 10.0.

These artifacts are quality-attribution artifacts, not deployable Watch SE
candidates. They are still around 2.0G because only 2-4 attention projection
weights were compressed.

## Prefix Gate

Reports:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-qk-int4-prefix-logits-diagnostics.json
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-o-int4-prefix-logits-diagnostics.json
```

Top-5 overlap against fp16:

```text
prefix   QK-only   O-only
1        5/5       5/5
2        5/5       5/5
4        5/5       5/5
8        5/5       5/5
12       4/5       4/5
16       5/5       5/5
18       5/5       5/5
```

The prefix gate does not distinguish QK-only and O-only strongly. Both are far
better than the failed Q/K/O-only result, which diverged at prefix 2.

## Teacher Smoke

Reports:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-qk-int4-teacher-smoke.json
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-o-int4-teacher-smoke.json
```

Result:

```text
candidate   generated IDs     token agreement   first token   peak RSS
QK-only     [1974, 10300]     1.0               236.64 ms     2157.64 MB
O-only      [1974, 10300]     1.0               247.36 ms     2162.17 MB
```

The single-prompt teacher smoke also does not separate them.

## Batch Gate

Reports:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-qk-int4-batch10-cap2.json
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-o-int4-batch10-cap2.json
artifacts/benchmarks/stateful-step-kv-256-fp16-batch10-cap2.json
```

Summary:

```text
candidate   avg agreement   succeeded   first-token avg   decode tok/s   peak RSS
fp16        0.90            10/10       216.32 ms         85.23          2450.44 MB
QK-only     0.90            10/10       213.83 ms         80.86          2186.86 MB
O-only      0.85            10/10       213.44 ms         79.12          2174.00 MB
```

QK-only matched fp16's per-prompt agreement profile exactly under this gate.
The only 0.0 prompt was `watch-utility-002`, which also emitted no tokens under
fp16.

O-only regressed `watch-utility-001`:

```text
fp16:   [354, 558]
QK:    [354, 558]
O:     [354, 1178]
```

## Interpretation

The result is more subtle than the earlier Q/K/O-only failure:

```text
QK-only: batch-level parity with fp16 under the current cap2 gate
O-only:  passes prefix/smoke but regresses one category-balanced prompt
Q/K/O:   fails immediately at prefix 2 and teacher smoke
```

This suggests the adjacent-layer Q/K/O failure is an interaction failure, not a
simple "all Q/K/O projections are individually unsafe" result.

QK-only is now a plausible small safe ingredient. O-only should remain
protected until a stronger calibrated quantization path exists.

## Next Consequence

The next high-value local candidate is not another whole-attention expansion.
It should be either:

```text
layer8-15 V-only + layer11-12 QK-only
```

or an activation-aware sensitivity scorer that decides whether Q/K expansion
should stay limited to layer11-12.

Follow-up note: the `layer8-15 V-only + layer11-12 QK-only` composition was
tested next and failed at prefix 2. See:

```text
docs/benchmarks/2026-05-30-stateful-step-layer8-15-v-layer11-12-qk-int4.md
```

Because QK-only compresses very few tensors, it is not itself a Watch SE
deployment strategy. Its value is directional: Q/K can be explored more
carefully, while O should not be widened based on the current evidence.
