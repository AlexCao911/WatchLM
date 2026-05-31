# Stateful Step V8 + Split-FFN Low-Risk Int4

Date: 2026-05-31

## Purpose

Test whether the risk-gated split-FFN candidates can be added to the known
stable `attentionV low8` ingredient.

Policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-importance-v8-ffn-gateup4-down4-lowrisk-int4.json
```

Selected int4 ops:

```text
attentionV: 8 ops, layers 6-13
ffnGateUp: 8 ops, layers 5, 6, 7, 9
ffnDown:   4 ops, layers 6, 7, 9, 10
```

## Conversion

Command:

```bash
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph stateful-step-kv \
  --context-tokens 256 \
  --source-mlpackage artifacts/coreml/real-minicpm5-stateful-step-kv-256/stateful-step-kv-256.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-stateful-step-importance-v8-ffn-gateup4-down4-lowrisk-int4.json \
  --output-dir artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-int4
```

Result:

```text
status:              succeeded
compression stage:   90.0s
selected int4 ops:   20
mlpackage bytes:     2,030,055,874
```

Generated artifacts:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos11-stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
```

Directory sizes:

```text
mlpackage:           1.9G
macOS compiled:      1.9G
watchOS 11 compiled: 1.9G
```

Both macOS and watchOS 11 Core ML compilation succeeded. The shell emitted a
non-fatal pyenv rehash warning during compile.

## Full Calibration Prefix Diagnostics

Command:

```bash
swift run WatchLMBenchmark \
  --runtime coreml \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --prefill artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc \
  --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json \
  --coreml-graph-interface stateful-step-kv \
  --diagnostics-top-k 5 \
  --diagnostics-prefix-lengths 1,2,4,8,12,18,32 \
  --context 256 \
  --policy-id importance-v8-ffn-gateup4-down4-lowrisk-int4 \
  --id stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-calibration-prefix-full12 \
  --output artifacts/benchmarks/stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-calibration-prefix-full12.json
```

Result:

```text
diagnostic points: 84/84 succeeded
compiled model bytes: 2,030,086,161
tokenizer bytes: 9,894,271
total selected artifact bytes: 2,039,980,432
```

## Sensitivity Against FP16

Command:

```bash
swift run WatchLMBenchmark \
  --sensitivity-baseline artifacts/benchmarks/stateful-step-kv-256-fp16-calibration-prefix-full12.json \
  --sensitivity-candidate artifacts/benchmarks/stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-calibration-prefix-full12.json \
  --output artifacts/benchmarks/stateful-step-kv-256-importance-v8-ffn-gateup4-down4-lowrisk-calibration-prefix-sensitivity-full12.json
```

Result:

```text
gate_ok:                         false
compared points:                 84
average prefill top-k overlap:   0.29
prefill top-1 agreement:         0.29
first zero-overlap prefix:       4 tokens
```

Overlap distribution:

```text
0/5 overlap: 60 points
5/5 overlap: 24 points
```

Gate failures:

```text
average prefill top-k overlap 0.29 is below 0.8 target
prefix 4 prefill overlap 0 is below 1 critical-prefix target
```

## Interpretation

This candidate is not acceptable. It proves the Core ML graph path and Swift
diagnostics can exercise the combined split-FFN policy, but it also shows the
current activation-risk heuristic is not sufficient for FFN int4 selection.

The important lesson is narrow:

```text
attentionV low8 remains the only full-calibration-passing low-bit ingredient.
adding ffnGateUp/ffnDown int4 with only activation-energy risk gates causes
large logits drift.
```

## Next Action

Do not widen FFN int4 from this candidate. The next useful experiments are:

```text
1. isolate ffnGateUp-only and ffnDown-only Core ML candidates against the same
   full12 sensitivity gate.
2. add a reconstruction-error term to the risk scorer before selecting more
   FFN layers.
3. consider FFN int8 or mixed int8/int4 before any further FFN int4 expansion.
```

This failure is useful because it moves the search away from blind low-energy
FFN quantization and toward reconstruction-aware FFN policy selection.
