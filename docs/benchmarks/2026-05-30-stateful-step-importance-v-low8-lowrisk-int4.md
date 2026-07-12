# Stateful Step Importance-Guided V-Low8 Low-Risk Int4

Date: 2026-05-30

## Scope

This note records the first larger V-only policy generated with the channel-risk
filter.

The previous low-risk candidate compressed four attention-V projections. This
candidate doubles that budget to eight V projections while still excluding:

```text
edge layers:                 0, 1, 2, 3, 20, 21, 22, 23
known layer4 outlier:        4
high top-column concentration: maxTopColumnEnergyFraction > 0.04
```

The result is a low-energy, low-concentration V-only set:

```text
attentionV layers 6, 7, 8, 9, 10, 11, 12, 13 -> int4
all other weights and KV state -> fp16
```

This is comparable in compression budget to the earlier manual layer8-15 V-only
experiment, but the selected layers come from calibration evidence rather than a
contiguous hand-picked window.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low8-lowrisk-int4.json
```

Policy id:

```text
stateful-step-importance-attention-v-low8-lowrisk-int4-rest-fp16
```

Selection evidence:

```text
source report: artifacts/benchmarks/minicpm5-activation-importance-cal12-groups.json
ranking:       lowest_component_activation_energy_with_channel_risk_filter
risk filter:   maxTopColumnEnergyFraction <= 0.04
excluded:      layers 0, 1, 2, 3, 4, 5, 20, 21, 22, 23
selected:
  layer6  energy 107708.40625, fraction 0.036717
  layer7  energy 116299.71875, fraction 0.018840
  layer8  energy 124729.48438, fraction 0.039045
  layer9  energy 125107.59375, fraction 0.008810
  layer11 energy 141500.93750, fraction 0.013403
  layer10 energy 160141.04688, fraction 0.010338
  layer12 energy 162804.84375, fraction 0.011303
  layer13 energy 169316.93750, fraction 0.012733
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos11-stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4/conversion-report.json
```

Size:

```text
mlpackage bytes: 2,157,454,606
artifact dirs:   2.0G mlpackage, 2.0G macOS compiled, 2.0G watchOS 11 compiled
```

Compression audit:

```text
int4 selected ops: 8
selected component: attentionV
selected layers:    6, 7, 8, 9, 10, 11, 12, 13
rejected ops:       1474
```

This is still not a Watch SE deployable artifact. Doubling V-only compression
from four tensors to eight tensors reduces the package by only about 2.36 MB
relative to the low-risk V-low4 artifact.

## Build Gates

Conversion:

```text
compress_coreml_weights_mixed: succeeded in 69.96s
```

macOS compile:

```text
succeeded
```

watchOS 11 compile:

```text
succeeded
```

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4-prefix-logits-diagnostics.json
```

Sensitivity report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4-sensitivity.json
```

Compared with fp16:

```text
gate_ok: true
average prefill top-k overlap: 0.94
prefill top-1 agreement: 1.0

prefix 1:  prefill overlap 5/5, decode overlap 4/5
prefix 2:  prefill overlap 5/5, decode overlap 5/5
prefix 4:  prefill overlap 4/5, decode overlap 5/5
prefix 8:  prefill overlap 5/5, decode overlap 5/5
prefix 12: prefill overlap 5/5, decode overlap 5/5
prefix 16: prefill overlap 5/5, decode overlap 4/5
prefix 18: prefill overlap 4/5, decode overlap 5/5
```

The gate passes and top-1 remains stable. Compared with the manual layer8-15
V-only policy, the top-5 drift is similar but selected by measured activation
energy/concentration rather than a contiguous window.

## Teacher Batch Gate

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4-batch10-cap2.json
```

Result:

```text
prompts:                 10/10 succeeded
average token agreement: 0.9
first token avg:         214.01 ms
decode throughput:       82.47 tok/s
peak RSS:                2186.58 MB
thermal states:          nominal
```

The agreement profile again matches fp16. The only 0.0 prompt is still
`watch-utility-002`, which also fails under fp16 because the runtime stops
without emitting the teacher EOS token.

## Interpretation

This is positive evidence for the calibrated V-only path:

```text
V-only can expand from four to eight selected layers while preserving current
Swift quality gates.
```

It is also negative evidence for deployability:

```text
V-only compression alone is too small a part of MiniCPM5-1B to solve Watch SE
package or runtime memory limits.
```

The next useful experiment should not keep widening V forever. The evidence now
suggests:

```text
1. V-only is a safe compression ingredient.
2. FFN remains unsafe under current Core ML int8/int4 compression.
3. The next deploy-size move needs a new compression surface:
   grouped-channel/sensitive-kmeans Core ML, activation-weighted reconstruction,
   or another tensor family with measured safety.
```
