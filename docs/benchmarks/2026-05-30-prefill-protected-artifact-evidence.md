# Prefill Protected Artifact Evidence

## Scope

This note records the first real context-16 prefill-KV artifact generated from the fp16-attention protected policy.

It is separate from the policy-schema note. This file is only about the produced artifact, watchOS compilation, and Swift/Core ML diagnostic result.

## Artifact

Command:

```text
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py --graph prefill-kv --source-mlpackage artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage --compression mixed --precision-policy tools/conversion/mixed-precision-policy-prefill-kv-protected.json --output-dir artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected
```

Result:

```text
mlpackagePath: artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected/prefill-kv-16-mixed.mlpackage
mlpackageBytes: 1241841784
elapsedSeconds: 57.43
```

Selector audit:

```text
int8 selected ops: 71
int8 selected components: embedding, ffn, lmHead
int4 selected ops: 3
int4 selected layer: 12 FFN
attention Q/K/O/V: fp16
kvCache policy: fp16
```

## watchOS Compile

Command:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected/prefill-kv-16-mixed.mlpackage artifacts/coreml/compiled-watchos-prefill-kv-16-prefill-protected --platform watchOS --deployment-target 10.0
```

Output:

```text
artifacts/coreml/compiled-watchos-prefill-kv-16-prefill-protected/prefill-kv-16-mixed.mlmodelc/coremldata.bin
```

## Swift Diagnostic

Added an env-gated Swift diagnostic test:

```text
coreMLPrefillDecodeDiagnosticsCanRunLocalMiniCPMPrefillProtectedArtifacts
```

Command:

```text
WATCHLM_RUN_REAL_COREML_TESTS=1 swift test --filter coreMLPrefillDecodeDiagnosticsCanRunLocalMiniCPMPrefillProtectedArtifacts
```

Output:

```text
WATCHLM_PREFILL_PROTECTED_DIAGNOSTIC protected-prefill-fp16-decode prefill=[416, 242, 1974, 359, 1] decode=[4245, 826, 11420, 5018, 2793] decodeMargin=0.101562 | protected-prefill-int8-decode prefill=[416, 242, 1974, 359, 1] decode=[4245, 826, 11420, 5018, 2793] decodeMargin=0.117188
```

The test passed in 67.430 seconds on the host.

## Interpretation

The protected prefill artifact restores the expected decode top-1 `4245` with both fp16 and int8 decode graphs.

Compared with the earlier matrix:

```text
int8 prefill -> int8 decode: 826
protected prefill -> int8 decode: 4245
fp16 prefill -> int8 decode: 4245
```

This makes the protected prefill policy a better next candidate than global int8 prefill for SE2/SE3 experiments. It is still too large to declare usable on watch hardware, but it gives us a concrete quality-preserving direction: keep prefill attention/KV fidelity high, then search FFN/lm_head compression and decode-side optimization separately.
