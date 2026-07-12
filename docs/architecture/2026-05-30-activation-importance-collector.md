# Activation Importance Collector

Date: 2026-05-30

## Scope

This note records the first PyTorch-side activation importance collector for
MiniCPM5-1B quantization search.

The collector is an offline conversion tool. It does not replace the Swift
watch inference chain. Its job is to produce evidence for mixed precision
policies before Core ML export.

## Files

```text
tools/conversion/collect-activation-importance.py
test/activationImportanceCli.test.js
```

The collector consumes:

```text
tools/benchmark/fixtures/calibration-prompts.json
```

and emits an importance report with:

```text
schemaVersion
sourceModelId
calibration suite summary
collection mode and statistic
target components
component summary
layer summary
per-module activation energy summaries
```

## Statistic

The initial statistic is:

```text
sum_input_activation_squared_by_column
```

For each collected module, the forward pre-hook sees the module input tensor,
flattens batch/sequence dimensions, and accumulates:

```text
sum(x[column]^2)
```

This is deliberately close to the practical imatrix intuition: columns with
larger input activation energy have more influence on downstream error when
their weights are perturbed.

## Component Mapping

Module names are mapped into the same policy components already used by the
Core ML conversion policies:

```text
attentionQKO
attentionV
ffn
embedding
lmHead
norms
```

This keeps the importance report directly useful for future policy generation.

## Usage

Dry-run schema check:

```bash
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --dry-run \
  --output artifacts/benchmarks/minicpm5-activation-importance-dry-run.json
```

Full collection, when local weights are available:

```bash
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --cache-dir artifacts/hf/MiniCPM5-1B \
  --quiet \
  --output artifacts/benchmarks/minicpm5-activation-importance.json
```

## First Real Smoke Collection

After the schema was added, the collector was run against the local
MiniCPM5-1B safetensors with one calibration prompt:

```bash
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --cache-dir artifacts/hf/MiniCPM5-1B \
  --max-prompts 1 \
  --top-columns 8 \
  --device cpu \
  --output artifacts/benchmarks/minicpm5-activation-importance-cal1.json
```

The run loaded the local 2.0 GB safetensors checkpoint and produced a report
with 218 module summaries in about 2-3 seconds after model load.

This proves the collector can execute the real model path locally, not only the
dry-run schema path.

The collector was then run over the full 12-prompt calibration suite:

```bash
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --cache-dir artifacts/hf/MiniCPM5-1B \
  --top-columns 8 \
  --device cpu \
  --quiet \
  --output artifacts/benchmarks/minicpm5-activation-importance-cal12.json
```

The full suite produced 218 module summaries and component/layer aggregates in
about 5 seconds after model load. This is now a usable input for policy
candidate generation.

## Current Limitations

The full 12-prompt report is still calibration evidence, not final quality
evidence. Any policy generated from it must still pass Core ML conversion and
the Swift quantization sensitivity scorer.

`norms` activation energy is also much larger than linear-layer energy and
should not be compared directly against Q/K/V/FFN totals for int4 selection.
Norms remain protected by policy. The useful ranking target is primarily within
linear module families and layers.

The next step is to use the full report to generate a small set of
importance-guided Core ML mixed precision candidates.
