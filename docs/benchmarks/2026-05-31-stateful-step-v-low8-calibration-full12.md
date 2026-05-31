# Stateful Step V-Low8 Full Calibration Prefix Gate

Date: 2026-05-31

## Purpose

After adding Swift `--calibration-prompts`, the first smoke used only two
calibration prompts. This run expands the same FP16-vs-candidate comparison to
the full 12-prompt calibration suite and all fixed prefix lengths.

The goal is to confirm whether the low-risk V-attention low8 policy remains
stable across the full calibration gate before using it as a safe ingredient in
larger mixed policies.

## Artifacts

Baseline:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256/stateful-step-kv-256.mlmodelc
```

Candidate:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc
```

Tokenizer:

```text
artifacts/hf/MiniCPM5-1B/tokenizer.json
```

Reports:

```text
artifacts/benchmarks/stateful-step-kv-256-fp16-calibration-prefix-full12.json
artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-full12.json
artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-sensitivity-full12.json
```

## Commands

Baseline:

```bash
swift run WatchLMBenchmark \
  --runtime coreml \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --prefill artifacts/coreml/compiled-macos-stateful-step-kv-256/stateful-step-kv-256.mlmodelc \
  --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json \
  --coreml-graph-interface stateful-step-kv \
  --diagnostics-top-k 5 \
  --diagnostics-prefix-lengths 1,2,4,8,12,18,32 \
  --context 256 \
  --policy-id stateful-step-kv-256-fp16 \
  --id stateful-step-kv-256-fp16-calibration-prefix-full12 \
  --output artifacts/benchmarks/stateful-step-kv-256-fp16-calibration-prefix-full12.json
```

Candidate:

```bash
swift run WatchLMBenchmark \
  --runtime coreml \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --prefill artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-attention-v-low8-lowrisk-int4/stateful-step-kv-256-mixed.mlmodelc \
  --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json \
  --coreml-graph-interface stateful-step-kv \
  --diagnostics-top-k 5 \
  --diagnostics-prefix-lengths 1,2,4,8,12,18,32 \
  --context 256 \
  --policy-id stateful-step-importance-attention-v-low8-lowrisk-int4-rest-fp16 \
  --id stateful-step-kv-256-low8-lowrisk-calibration-prefix-full12 \
  --output artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-full12.json
```

Sensitivity:

```bash
swift run WatchLMBenchmark \
  --sensitivity-baseline artifacts/benchmarks/stateful-step-kv-256-fp16-calibration-prefix-full12.json \
  --sensitivity-candidate artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-full12.json \
  --output artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-sensitivity-full12.json
```

## Result

Diagnostics:

```text
baseline:  84/84 succeeded
candidate: 84/84 succeeded
```

Sensitivity:

```text
gate_ok:                         true
compared points:                 84
average prefill top-k overlap:   0.90
prefill top-1 agreement:         0.96
gate failures:                   none
```

Overlap distribution:

```text
3/5 overlap:  4 points
4/5 overlap: 36 points
5/5 overlap: 44 points
```

There were no prefix points where the candidate top-1 token differed from the
baseline top-1 token.

## Interpretation

The low-risk V-attention low8 policy remains stable on the full calibration
prefix gate. This is stronger evidence than the earlier two-prompt smoke and
supports treating this V policy as a safe ingredient in the next mixed policy.

It still does not solve Watch SE2/SE3 deployability by itself. V-only
compression saves too little model size. The next experiment should combine
this stable V ingredient with a community-shaped mixed 4/8/fp16 policy and an
activation-weighted rejection gate before Core ML export.
