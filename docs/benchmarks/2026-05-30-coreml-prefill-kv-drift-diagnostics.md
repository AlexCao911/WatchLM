# Core ML Prefill KV Drift Diagnostics

## Scope

This note records the first Swift-side logits diagnostic for the context-16 int8 Core ML chain.

The goal was to locate why `en-short-001` generated `[416,826]` in the Swift benchmark while the context-aligned PyTorch split-graph teacher expected `[416,4245]`.

## Diagnostic Added

Added `CoreMLPrefillDecodeDiagnostics` in Swift. It runs the same Core ML prefill/decode graph path as the runtime and exposes:

- encoded prompt token IDs
- prefill logits top-k
- first decode logits top-k after feeding decode with Core ML prefill KV

This is intentionally lower-level than `RuntimeBenchmarkRunner`, which only records selected generated tokens.

## Smoke Verification

```text
swift test --filter coreMLPrefillDecodeDiagnosticsExposePrefillAndDecodeTopK: 1 test passed
```

The smoke layered graph returns:

```text
prefill top token: 5
first decode top token: 6
```

## Real Int8 Evidence

Command:

```text
WATCHLM_RUN_REAL_COREML_TESTS=1 swift test --filter coreMLPrefillDecodeDiagnosticsCanRunLocalRealMiniCPMInt8Artifacts
```

Output:

```text
WATCHLM_REAL_INT8_DIAGNOSTIC en-short-001 prefill=[416, 242, 1974, 359, 861] decode=[826, 4245, 11420, 5018, 4183]
```

Earlier PyTorch-KV decode validation for the same prompt showed:

```text
torchTopK:  [4245,826,11420,5018,2793,4183,6971,2042,1903,3402]
coremlTopK: [4245,826,11420,5018,2793,4183,6971,2042,1903,3402]
top1Matches: true
```

## Interpretation

- Swift tokenizer parity is not the issue.
- Context-aligned split-graph teacher is not the issue.
- The standalone decode graph is not the issue when fed PyTorch KV.
- The prefill selected token is correct in the Swift/Core ML `.all` execution path.
- The first decode step flips top-1 only after using Core ML prefill KV.

The strongest current hypothesis is that prefill KV drift from the int8 prefill graph is enough to swap the first two decode logits for this prompt. The next diagnostic should compare decode logits using:

```text
PyTorch prefill KV -> Core ML decode
Core ML int8 prefill KV -> Core ML decode
Core ML fp16 prefill KV -> Core ML decode
```

If fp16 prefill KV restores `[4245,826,...]`, the optimization work should protect KV-producing attention paths more aggressively than the current global int8 baseline.
