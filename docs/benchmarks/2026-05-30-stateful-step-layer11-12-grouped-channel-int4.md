# Stateful Step Layer11-12 Grouped-Channel Attention Int4

Date: 2026-05-30

## Scope

This note records a prior-led retry of the failed layer11-12 attention int4
window. The hypothesis came from Core ML's palettization guidance: per-tensor
LUTs can produce high approximation error for large matrices, while
`per_grouped_channel` creates multiple LUTs across channel groups.

The goal was not to widen another layer window. It was to test whether the
layer11-12 failure was caused by coarse per-tensor k-means palettization.

## Policy Support Added

The conversion policy now accepts an optional `int4Compression` block:

```text
method:                palettization
mode:                  kmeans
granularity:           per_tensor | per_grouped_channel
groupSize:             positive integer
enablePerChannelScale: boolean
clusterDim:            positive integer
numKMeansWorkers:      positive integer
weightThreshold:       non-negative integer or null
```

The converter uses Core ML weight metadata to build explicit `op_name_configs`,
so we can keep selective compression while still using the newer
`OpPalettizerConfig` settings.

## Per-Channel Scale Attempt

Policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-int4-grouped-channel.json
```

Settings:

```text
granularity:           per_grouped_channel
groupSize:             16
enablePerChannelScale: true
selected ops:          layer11-12 attention Q/K/O/V
```

Result:

```text
status: failed during Core ML compile/save
error:  mps.dequantize operand expected quantized values but saw tensor<1xf16>
```

Interpretation:

```text
The current coremltools 9.0 + local Xcode compiler path cannot promote this
specific grouped-channel + per-channel-scale MLProgram into a valid executable
artifact. The feature is useful as a prior, but this exact setting is blocked
by compiler compatibility before quality can be measured.
```

## No-Scale Grouped-Channel Attempt

Policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-int4-grouped-channel-noscale.json
```

Settings:

```text
granularity:           per_grouped_channel
groupSize:             16
enablePerChannelScale: false
selected ops:          layer11-12 attention Q/K/O/V
```

Artifacts:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-layer11-12-attention-int4-grouped-channel-noscale/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-layer11-12-attention-int4-grouped-channel-noscale/stateful-step-kv-256-mixed.mlmodelc
```

Conversion:

```text
status: succeeded
selected ops: 8
selected layer11 ops: q/k/v/o
selected layer12 ops: q/k/v/o
mlpackage bytes: 2,140,955,408
compiled artifact: 2.0 GB
```

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-int4-grouped-channel-noscale-prefix-logits-diagnostics.json
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
  candidate [242, 84, 316, 33, 1688]
  overlap:  0/5

prefix 8:
  fp16      [3732, 1674, 242, 1494, 2790]
  candidate [702, 282, 29365, 22855, 39272]
  overlap:  0/5

prefix 12:
  fp16      [36734, 2319, 2242, 3229, 2218]
  candidate [4726, 26459, 11661, 10483, 6517]
  overlap:  0/5

prefix 16:
  fp16      [280, 285, 691, 450, 7287]
  candidate [88, 75115, 5020, 9241, 54140]
  overlap:  0/5

prefix 18:
  fp16      [1974, 591, 343, 416, 2452]
  candidate [5, 49, 24, 1307, 45050]
  overlap:  0/5
```

## Teacher Smoke

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-layer11-12-attention-int4-grouped-channel-noscale-teacher-smoke.json
```

Result:

```text
generated IDs: [5, 5]
teacher IDs:   [1974, 10300]
token agreement: 0.0
first token: 9289.18 ms
decode throughput: 1.05 tok/s
peak RSS: 2153.67 MB
```

## Interpretation

Grouped-channel palettization without per-channel scale compiles, but it does
not recover quality. The candidate diverges at prefix 2, same as the original
layer11-12 attention window.

This rules out a narrow explanation: the layer11-12 failure is not merely caused
by a single per-tensor LUT being too coarse. It is more likely caused by
projection-level sensitivity, adjacent-layer accumulation, or the need for
activation/importance-aware scaling before compression.

Next experiments should therefore be structural:

```text
Q/K/O-only versus V-only attribution
activation-aware sensitivity scoring
calibrated PyTorch-side palettization or AWQ-style scaling
```
