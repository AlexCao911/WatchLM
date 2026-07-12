# Swift Calibration Prompt Diagnostics

Date: 2026-05-30

## Purpose

The Swift benchmark CLI can now consume the quantization calibration prompt suite directly. This keeps calibration prompts, prefix diagnostics, and sensitivity reports on the same Swift/Core ML path that will be used for watch deployment decisions.

## Runtime Change

`WatchLMBenchmark` accepts:

```bash
--calibration-prompts tools/benchmark/fixtures/calibration-prompts.json
```

The command loads and validates `QuantizationCalibrationSuite`, converts it to `RuntimeBenchmarkPromptSuite`, and then reuses the existing prompt selection, prompt limit, max-new-token cap, teacher sidecar, Core ML diagnostics, and sensitivity comparison flow.

`RuntimeBenchmarkPromptSuite` now distinguishes required categories from supported categories. `stop_sequence` is supported so calibration diagnostics can include stop-boundary prompts without forcing the smaller smoke benchmark suite to require that category.

## Smoke Run

Baseline:

```bash
swift run WatchLMBenchmark \
  --runtime coreml \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --prefill artifacts/coreml/compiled-macos-stateful-step-kv-256/stateful-step-kv-256.mlmodelc \
  --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json \
  --coreml-graph-interface stateful-step-kv \
  --diagnostics-top-k 5 \
  --diagnostics-prefix-lengths 1,2,4,8 \
  --prompt-limit 2 \
  --context 256 \
  --policy-id stateful-step-kv-256-fp16 \
  --id stateful-step-kv-256-fp16-calibration-prefix-smoke \
  --output artifacts/benchmarks/stateful-step-kv-256-fp16-calibration-prefix-smoke.json
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
  --diagnostics-prefix-lengths 1,2,4,8 \
  --prompt-limit 2 \
  --context 256 \
  --policy-id stateful-step-importance-attention-v-low8-lowrisk-int4-rest-fp16 \
  --id stateful-step-kv-256-low8-lowrisk-calibration-prefix-smoke \
  --output artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-smoke.json
```

Sensitivity:

```bash
swift run WatchLMBenchmark \
  --sensitivity-baseline artifacts/benchmarks/stateful-step-kv-256-fp16-calibration-prefix-smoke.json \
  --sensitivity-candidate artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-smoke.json \
  --output artifacts/benchmarks/stateful-step-kv-256-low8-lowrisk-calibration-prefix-sensitivity-smoke.json
```

## Result

- Baseline diagnostics: 8/8 succeeded
- Candidate diagnostics: 8/8 succeeded
- Sensitivity gate: pass
- Compared points: 8
- Average prefill top-k overlap: 0.95
- Prefill top-1 agreement: 1.0

## Interpretation

This does not prove deployment readiness. It proves the calibration prompt suite now drives the real Swift/Core ML diagnostics path, so future quantization experiments can be compared against a stable prompt/prefix fixture instead of ad hoc benchmark prompts.

The low-risk V-attention low8 policy still looks quality-safe on this small prefix smoke, but it is not an aggressive enough size reduction by itself. The next strategy work should be guided by external quantization evidence and model-structure intuition before expanding experiments.
