# Swift Quantization Sensitivity Scorer

Date: 2026-05-30

## Scope

This note records the Swift-side scorer added after the layer8-15 V plus
layer11-12 QK composition failure.

It is intentionally separate from benchmark result notes. The purpose is to
define the reusable gate that decides whether a quantization candidate is worth
promoting to slower host/watch tests.

## Problem

The latest evidence shows that locally safe components do not compose
additively:

```text
layer8-15 V-only int4:
  matches the fp16 batch10 cap2 token-agreement profile

layer11-12 QK-only int4:
  matches the fp16 batch10 cap2 token-agreement profile

layer8-15 V-only + layer11-12 QK-only:
  collapses at prefix 2 with 0/5 prefill top-k overlap
```

Continuing to widen layer windows would be a blind search. The next step needs
a measurable sensitivity gate that catches early logit collapse before a
candidate is treated as an SE2 deployment candidate.

## Swift Contract

The scorer lives in `WatchLMCore`:

```text
Sources/ModelRuntime/Eval/QuantizationSensitivity.swift
```

It compares two diagnostics reports as ordered prompt/prefix points:

```text
baseline Core ML diagnostics
candidate Core ML diagnostics
```

Each point records:

```text
prompt id
category
language
prefix token count
prefill top-k logits
optional decode top-k logits
```

The scorer emits:

```text
per-prefix top-k overlap
prefill top-1 agreement
first zero-overlap prefix
pass/fail gate with explicit failure strings
```

The default gate is deliberately conservative:

```text
minimum average prefill top-k overlap: 0.8
critical prefix window: <= 4 tokens
minimum critical-prefix overlap: 1 token
```

This catches the prefix-2 collapse pattern that appeared in the failed
composition experiment.

## Benchmark CLI

The benchmark executable now has a diagnostics comparison mode:

```bash
swift run WatchLMBenchmark \
  --sensitivity-baseline path/to/fp16-diagnostics.json \
  --sensitivity-candidate path/to/candidate-diagnostics.json \
  --output path/to/sensitivity-report.json
```

The input files are normal `CoreMLDiagnosticsReport` JSON files produced by the
existing diagnostics path:

```bash
swift run WatchLMBenchmark \
  --runtime coreml \
  --prefill path/to/model.mlpackage \
  --tokenizer path/to/tokenizer.json \
  --coreml-graph-interface stateful-step-kv \
  --diagnostics-top-k 5 \
  --diagnostics-prefix-lengths 1,2,4,8,12,18 \
  --output path/to/diagnostics.json
```

## How This Connects To Community Priors

The scorer is not a replacement for AWQ, imatrix, GPTQ, SmoothQuant, or
Core ML calibration-data palettization. It is the first Swift-native runtime
gate that lets those priors become disciplined experiments.

The intended loop is:

```text
community / architecture prior
  -> choose a small tensor family or layer window
  -> export Core ML candidate
  -> produce baseline and candidate diagnostics
  -> run Swift sensitivity scorer
  -> only promote candidates that survive early-prefix drift
```

This keeps later experiments hypothesis-driven:

```text
AWQ prior:
  protect salient channels found by activation behavior

imatrix prior:
  use representative prompt statistics instead of data-free low-bit guesses

DS4 prior:
  compress the structurally safe bulk; keep decision-shaping tensors higher
  precision

Core ML prior:
  prefer grouped/calibrated palettization when per-tensor LUT error is high
```

## Current Limitations

This scorer uses logits diagnostics, not internal activation tensors. That
means it can reject bad candidates and rank visible drift, but it cannot yet
identify the exact channel outliers that an AWQ-style transform would protect.

The next higher-value addition is a calibration pass that collects either:

```text
per-tensor perturbation drift
```

or:

```text
activation / weight-channel statistics before Core ML export
```

The scorer should remain the Swift-side promotion gate even after those
calibrated policies exist.
