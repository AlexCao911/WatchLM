# Qwen Stateful-Step Context256 FP32-Compute Int8

Date: 2026-05-31

## Goal

Move the working Qwen context256 explicit-KV baseline into a single shared
stateful-step Core ML graph, while keeping the Qwen-required float32 compute
precision and using int8 weight storage.

This is the current Qwen Swift inference candidate for the Apple Watch path. It
is not yet a final SE2/SE3 deployment claim.

## Artifact

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlpackage
```

Conversion settings:

```text
model: Qwen/Qwen3-0.6B
graph: stateful-step-kv
context: 256
torch dtype: float16
Core ML compute precision: float32
compression: int8
layers: 28
KV heads: 8
head dimension: 128
```

Sizes:

```text
mlpackage: 598,433,320 bytes
tokenizer:  11,422,654 bytes
total:     609,855,974 bytes
du size:   571 MB
```

The matching uncompressed fp32-compute source graph is preserved in the same
artifact directory as `stateful-step-kv-256.mlpackage` for follow-up int4 and
mixed-compression experiments.

## Swift Smoke

Report:

```text
artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8-swift-smoke.json
```

Prompt:

```text
watch-utility-002
Qwen3 non-thinking chat template
max new tokens: 4
```

Observed output:

```text
generatedTokenIDs: [785, 1614, 9329, 374]
text: "The model asset is"
termination: maxTokens
```

Metrics:

```text
loadMs: 3714.747
firstTokenMs: 567.678
decodeStepMs: 15.968, 15.976, 15.725
averageDecodeTokensPerSecond: 62.93
peakResidentMemoryMB: 1581.16
thermal: nominal
```

Compared with the split explicit-KV int8 baseline, this keeps the same generated
tokens while reducing host peak RSS from about 2.74 GB to about 1.58 GB.

## Top-K Diagnostics

Report:

```text
artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8-topk.json
```

For the full 32-token prompt prefix:

```text
prefill top1: token 785, logit 22.296875
decode top1:  token 1614, logit 23.890625
```

This matches the trusted explicit-KV int8 baseline:

```text
prefill top1: token 785
decode top1:  token 1614
```

The older stateful-step int8 artifact converted from a float16-compute source
failed this same check by selecting EOS token `151645` at the full prompt. The
fix is therefore not a Swift sampler or tokenizer change; the Qwen stateful
route needs a float32-compute Core ML graph before weight compression.

## watchOS Compile

Compiled with:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile \
  artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlpackage \
  artifacts/coreml/compiled-watchos11-qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8 \
  --platform watchOS \
  --deployment-target 11.0
```

Result:

```text
artifacts/coreml/compiled-watchos11-qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlmodelc
compiled size: 571 MB
status: succeeded
```

## Conversion Script Fix

During this run, the conversion command spent several minutes in Hugging Face
snapshot checking because the script defaulted every model to the MiniCPM cache
directory unless `--cache-dir` was provided.

The conversion script now derives the default cache directory from `--model-id`
and supports `--local-files-only`. Future Qwen conversions should use:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --model-id Qwen/Qwen3-0.6B \
  --local-files-only \
  --torch-dtype float16 \
  --compute-precision float32 \
  --graph stateful-step-kv \
  --context-tokens 256 \
  --output-dir artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8 \
  --compression int8
```

## Interpretation

The current Qwen Swift/CoreML inference chain is real and coherent:

```text
Tokenizer -> stateful-step PrefillGraph -> Core ML state KV cache
-> DecodeGraph -> LogitsProcessor -> Sampler -> token text
```

The strongest candidate has shifted from split explicit-KV int8 to shared
stateful-step int8 with float32 compute. Remaining deployment work is now:

1. Compress this fp32-compute stateful source toward int4 or mixed int4 without
   losing the top-k gate.
2. Run the compiled artifact through the watchOS simulator/app bundle path.
3. Measure real SE2/SE3 device memory, first-token latency, decode speed, and
   jetsam/thermal behavior.
