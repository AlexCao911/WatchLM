# Local Artifact Retention

## Purpose

Keep the local WatchLM workspace small enough to continue Qwen/CoreML inference
work without losing the artifacts that are still useful for immediate testing,
comparison, or reproducible conclusions.

## Retained

The current local artifact set is intentionally limited to:

- `artifacts/hf/Qwen3-0.6B`
  - source checkpoint and tokenizer needed to regenerate Qwen Core ML artifacts
    and verify tokenizer parity.
- `artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-int4`
  - current Qwen int4 `stateful-step-kv` Core ML package used by Swift host
    inference.
- `artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp16`
  - Qwen fp16 diagnostic baseline retained because the int4 smoke drift also
    reproduces without int4 compression.
- `artifacts/coreml/compiled-watchos11-qwen3-0.6b-stateful-step-kv-256-int4`
  - current watchOS 11 compiled Qwen package for simulator/device deploy
    checks.
- `artifacts/hf/MiniCPM5-1B`
  - MiniCPM teacher/reference checkpoint and tokenizer for comparisons.
- `artifacts/coreml/real-minicpm5-stateful-step-kv-256-int4`
  - smallest representative MiniCPM Core ML baseline for memory/quality
    comparison.
- `artifacts/benchmarks`
  - small JSON evidence from prior runs. These are cheap to keep and preserve
    exact measured values.

## Removed

The following local artifacts were removed because they are either parked,
regenerable, or already summarized in docs:

- `artifacts/gguf`
  - official MiniCPM GGUF route is not the active Core ML path on this branch.
- `artifacts/coreml/qwen3-0.6b-stateful-kv-256-fp16`
  - failed Qwen `stateful-kv` diagnostic artifact. It converted, but the Swift
    generation path failed Core ML execution-plan build with error `-14`.
- `artifacts/coreml/compiled-macos-qwen3-0.6b-stateful-step-kv-256-fp16`
  - temporary compiled macOS cache for the Qwen fp16 diagnostic package.
- `artifacts/coreml/compiled-macos-stateful-step-kv-256-int4`
  - MiniCPM compiled cache; source package is enough for reference.
- `artifacts/coreml/compiled-watchos-stateful-step-kv-256-int4`
  - MiniCPM compiled cache; source package is enough for reference.
- `artifacts/coreml/compiled-watchos-stateful-step-kv-256-protected-no-int4`
  - oversized MiniCPM protected/no-int4 compiled cache.
- `artifacts/coreml/real-minicpm5-stateful-step-kv-256-protected-no-int4`
  - oversized MiniCPM protected/no-int4 package; results are already recorded
    and the artifact can be regenerated from the checkpoint.

## Size After Cleanup

Measured after cleanup on 2026-05-31:

```text
artifacts:        5.6 GB
artifacts/coreml: 2.2 GB
artifacts/hf:     3.4 GB
artifacts/tmp:    0 B
free disk:        about 65 GiB
```

This keeps the active Qwen watch path available while preserving one MiniCPM
baseline and the model checkpoints needed for reproducibility.
