# Calibration Prompt Suite

Date: 2026-05-30

## Scope

This note records the first WatchLM calibration prompt suite used to guide
MiniCPM5-1B quantization experiments.

It is separate from benchmark result notes. The suite is an input asset for
importance collection and sensitivity scoring, not a benchmark result by itself.

## Files

```text
tools/benchmark/fixtures/calibration-prompts.json
Sources/ModelRuntime/Eval/QuantizationCalibration.swift
tools/benchmark/calibrationPrompts.js
```

The Swift contract is authoritative for runtime-side consumption. The Node
validator exists so conversion and validation scripts can reject malformed
calibration assets before long-running model work starts.

## Contract

The suite fixes the current SE2 calibration target:

```text
model: openbmb/MiniCPM5-1B
tokenizer: openbmb/MiniCPM5-1B
context: 256
prompt format: minicpm5-chat-template-no-think
prefix sweep: 1, 2, 4, 8, 12, 18, 32
```

Required categories:

```text
zh_short_instruction
en_short_instruction
watch_utility
code_small_fix
stop_sequence
safety_refusal
```

Every rendered prompt uses the MiniCPM no-think assistant prefix:

```text
<|im_start|>assistant
<think>

</think>
```

## Why This Matters

The previous manual quantization search found useful local clues, but also
showed that individually safe tensor families can fail when combined. The
calibration suite is the first shared input for an evidence loop:

```text
calibration prompts
  -> fp16 diagnostics
  -> activation / perturbation importance
  -> mixed precision policy
  -> Core ML candidate
  -> Swift quantization sensitivity scorer
```

This makes each future candidate explainable. A tensor should be compressed
because calibration evidence marks it low-impact, not because it happened to be
the next layer in a sweep.

## Current Limitations

The suite is intentionally small: 12 prompts, two per required category. It is
large enough to catch prompt-template and early-prefix collapse issues, but not
large enough to estimate full model quality.

The next step is to add a PyTorch-side activation importance collector that
consumes this exact suite and emits per-module input activation statistics.
