# Prefill KV Precision Matrix

## Scope

This note records a Swift/Core ML diagnostic matrix for the context-16 MiniCPM5 split graph.

The goal was to isolate whether the `en-short-001` second-token drift comes from the decode graph or from the KV tensors produced by the prefill graph.

## Diagnostic

Added an env-gated Swift test:

```text
coreMLPrefillDecodeDiagnosticsCanCompareLocalMiniCPMPrefillPrecisionArtifacts
```

It runs the same prompt through four graph pairings:

```text
fp16 prefill -> fp16 decode
fp16 prefill -> int8 decode
int8 prefill -> fp16 decode
int8 prefill -> int8 decode
```

Command:

```text
WATCHLM_RUN_REAL_COREML_TESTS=1 swift test --filter coreMLPrefillDecodeDiagnosticsCanCompareLocalMiniCPMPrefillPrecisionArtifacts
```

## Evidence

Output:

```text
WATCHLM_PREFILL_PRECISION_DIAGNOSTIC fp16-prefill-fp16-decode prefill=[416, 242, 1974, 359, 861] decode=[4245, 826, 11420, 5018, 2793] decodeMargin=0.039062 | fp16-prefill-int8-decode prefill=[416, 242, 1974, 359, 861] decode=[4245, 826, 11420, 5018, 2793] decodeMargin=0.070312 | int8-prefill-fp16-decode prefill=[416, 242, 1974, 359, 861] decode=[826, 4245, 11420, 5018, 2793] decodeMargin=0.054688 | int8-prefill-int8-decode prefill=[416, 242, 1974, 359, 861] decode=[826, 4245, 11420, 5018, 4183] decodeMargin=0.023438
```

The test passed in 104.137 seconds on the host.

## Interpretation

The result isolates the drift to prefill KV precision:

```text
fp16 prefill -> int8 decode: decode top1 is 4245
int8 prefill -> fp16 decode: decode top1 is 826
```

That means the decode graph can preserve the teacher top1 under int8 weights when fed fp16 prefill KV. The top1 flips as soon as the prefill graph is int8, even with an fp16 decode graph.

The current global int8 policy is therefore too aggressive for the prefill graph's KV-producing path. The next quantization pass should keep the prefill KV path at fp16 or a more conservative precision while continuing to test int8/int4 on less sensitive FFN regions.

## Implication for SE2/SE3

This does not yet solve the size problem. It narrows the optimization direction:

- Avoid promoting global int8 prefill as the fidelity baseline.
- Keep fp16 or higher-fidelity quantization around prefill attention/KV-producing tensors.
- Continue using int8 decode as a viable candidate, since fp16 prefill plus int8 decode preserved this prompt's top1.
- Measure the size and latency cost of protecting prefill KV paths before moving to larger context variants or physical SE2/SE3 tests.
