# Qwen Explicit-KV Context256 Int8 Smoke

Date: 2026-05-31

## Goal

Move the Qwen explicit-KV path from context16 graph smoke to a context256
runtime smoke that can carry the real Qwen chat template without truncating the
prompt.

This is still a host/macOS Core ML benchmark, not an Apple Watch deployment
claim.

## Artifacts

Generated:

```text
artifacts/coreml/qwen3-0.6b-prefill-kv-256-fp16-load-fp32-compute-int8/prefill-kv-256-int8.mlpackage
artifacts/coreml/qwen3-0.6b-decode-256-fp16-load-fp32-compute-int8/decode-256-int8.mlpackage
```

Sizes:

```text
prefill: 598,295,262 bytes
decode:  598,453,462 bytes
tokenizer: 11,422,654 bytes
total payload: 1,208,171,378 bytes
```

Conversion settings:

```text
model: Qwen/Qwen3-0.6B
graph: prefill-kv + decode
context: 256
torch dtype: float16
Core ML compute precision: float32
compression: int8
```

## Prefill Gate

Report:

```text
artifacts/benchmarks/qwen3-0.6b-prefill-kv-256-int8-validate.json
```

Result:

```text
topKAgreement: 9/10
top1Matches: true
meanAbsoluteError: 0.0972961709
maxAbsoluteError: 0.484375
gate: pass
```

This matches the context16 int8 bridge quality pattern while using the real
context256 graph shape.

## Swift Runtime Smoke

Manual-argument report:

```text
artifacts/benchmarks/qwen3-0.6b-explicit-kv-256-int8-swift-smoke.json
```

Manifest-driven report:

```text
artifacts/benchmarks/qwen3-0.6b-explicit-kv-256-int8-manifest-swift-smoke.json
```

Prompt:

```text
watch-utility-002
Qwen3 non-thinking chat template
max new tokens: 4
```

Output:

```text
generatedTokenIDs: [785, 1614, 9329, 374]
text: "The model asset is"
```

Manual report metrics:

```text
firstTokenMs: 203.39
decodeStepMs: 64.017, 15.224, 14.527
averageDecodeTokensPerSecond: 31.99
peakResidentMemoryMB: 2762.31
loadMs: 6010.954
```

Manifest report metrics:

```text
firstTokenMs: 214.216
averageDecodeTokensPerSecond: 30.59
peakResidentMemoryMB: 2736.05
```

The manifest run used a temporary asset root with symlinks to the generated
artifacts, so the size fields in that report reflect symlink sizes. The runtime
load and generation path used the real mlpackages.

## Interpretation

Context256 explicit-KV Qwen inference now works through the Swift
Tokenizer -> PrefillGraph -> KV cache -> DecodeGraph -> LogitsProcessor ->
Sampler chain.

This does not solve Apple Watch SE2/SE3 deployment yet. The payload is still
about 1.21 GB and the split prefill/decode host RSS is about 2.7 GB. The next
deployment-oriented step is to move this trusted context256 int8 baseline into a
single shared/stateful graph, then re-open layer-aware int4 experiments against
that Qwen-specific baseline.
