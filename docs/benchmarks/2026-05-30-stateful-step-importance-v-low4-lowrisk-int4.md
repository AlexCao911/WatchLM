# Stateful Step Importance-Guided V-Low4 Low-Risk Int4

Date: 2026-05-30

## Scope

This note records the first policy candidate generated from channel/group risk
statistics.

The previous importance-guided V-low4 policy selected layers 5, 6, 7, and 8 by
lowest total attention-V activation energy after protecting edge layers and the
layer4 outlier. This low-risk variant keeps the same size budget but also
filters layers whose attention-V `topColumnEnergyFraction` exceeds 0.04.

The result replaces layer5 with layer9:

```text
previous V-low4:  attentionV layers 5, 6, 7, 8
low-risk V-low4: attentionV layers 6, 7, 8, 9
```

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-lowrisk-int4.json
```

Policy id:

```text
stateful-step-importance-attention-v-low4-lowrisk-int4-rest-fp16
```

Selection evidence:

```text
source report: artifacts/benchmarks/minicpm5-activation-importance-cal12-groups.json
ranking:       lowest_component_activation_energy_with_channel_risk_filter
risk filter:   maxTopColumnEnergyFraction <= 0.04
excluded:      layers 0, 1, 2, 3, 4, 5, 20, 21, 22, 23
selected:
  layer6 energy 107708.40625, fraction 0.036717
  layer7 energy 116299.71875, fraction 0.018840
  layer8 energy 124729.48438, fraction 0.039045
  layer9 energy 125107.59375, fraction 0.008810
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos11-stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4/conversion-report.json
```

Size:

```text
mlpackage bytes: 2,159,812,818
artifact dirs:   2.0G mlpackage, 2.0G macOS compiled, 2.0G watchOS 11 compiled
```

Compression audit:

```text
int4 selected ops: 4
selected component: attentionV
selected layers:    6, 7, 8, 9
rejected ops:       1478
```

The size is unchanged in practice because the candidate still compresses only
four V projection tensors.

## Build Gates

Conversion:

```text
compress_coreml_weights_mixed: succeeded in 73.33s
```

watchOS compile:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile \
  artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4/stateful-step-kv-256-mixed.mlpackage \
  artifacts/coreml/compiled-watchos11-stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4 \
  --platform watchOS \
  --deployment-target 11.0
```

Result:

```text
succeeded
```

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4-prefix-logits-diagnostics.json
```

Sensitivity report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4-sensitivity.json
```

Compared with fp16:

```text
gate_ok: true
average prefill top-k overlap: 0.94
prefill top-1 agreement: 1.0

prefix 1:  prefill overlap 5/5, decode overlap 5/5
prefix 2:  prefill overlap 5/5, decode overlap 5/5
prefix 4:  prefill overlap 4/5, decode overlap 5/5
prefix 8:  prefill overlap 5/5, decode overlap 5/5
prefix 12: prefill overlap 5/5, decode overlap 5/5
prefix 16: prefill overlap 4/5, decode overlap 5/5
prefix 18: prefill overlap 5/5, decode overlap 4/5
```

This exactly matches the previous V-low4 sensitivity profile.

## Teacher Batch Gate

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low4-lowrisk-int4-batch10-cap2.json
```

Result:

```text
prompts:                 10/10 succeeded
average token agreement: 0.9
first token avg:         215.49 ms
decode throughput:       86.28 tok/s
peak RSS:                2177.25 MB
thermal states:          nominal
```

The agreement profile again matches fp16. The only 0.0 prompt is still
`watch-utility-002`, which also fails under fp16 because the runtime stops
without emitting the teacher EOS token.

## Interpretation

The channel-risk filter is useful, but not because it improved this four-tensor
candidate's visible quality score. Its value is that it gives a principled way
to avoid concentrated channel outliers before building larger policies:

```text
layer5 V had low total energy but higher top-column concentration
layer9 V had similar total energy and much lower concentration
both choices pass current quality gates at this tiny compression budget
```

The next experiment should use the same risk signal at a larger budget, where
bad layer choices are more likely to accumulate:

```text
expand V-only by low energy + low concentration
avoid same-layer Q/K/V composition until a stronger reconstruction metric exists
continue protecting FFN under current Core ML int8/int4 compression
```
