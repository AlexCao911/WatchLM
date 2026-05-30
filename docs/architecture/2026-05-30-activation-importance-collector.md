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
  --output artifacts/benchmarks/minicpm5-activation-importance.json
```

## Current Limitations

This commit proves the report schema and dry-run path. It does not yet include
a committed full MiniCPM5 activation report, because that requires loading the
local 1B model weights.

The next step is to run the collector against local weights, then use the
result to generate a small set of importance-guided Core ML mixed precision
candidates.
