# Prefill Protected No-Int4 Benchmark

## Scope

This note records the no-int4 protected prefill experiment.

The previous protected policy restored `en-short-001` and `watch-utility-001`, but regressed `code-fix-001`. This experiment removes the layer-12 FFN int4 override while keeping attention Q/K/O/V and KV cache policy at fp16.

## Artifact

Command:

```text
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py --graph prefill-kv --source-mlpackage artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage --compression mixed --precision-policy tools/conversion/mixed-precision-policy-prefill-kv-protected-no-int4.json --output-dir artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected-no-int4
```

Result:

```text
mlpackagePath: artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected-no-int4/prefill-kv-16-mixed.mlpackage
mlpackageBytes: 1252480141
elapsedSeconds: 51.915
```

Selector audit:

```text
int8 selected ops: 74
int8 selected components: embedding, ffn, lmHead
int4 selected ops: 0
attention Q/K/O/V: fp16
kvCache policy: fp16
```

## watchOS Compile

Command:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected-no-int4/prefill-kv-16-mixed.mlpackage artifacts/coreml/compiled-watchos-prefill-kv-16-prefill-protected-no-int4 --platform watchOS --deployment-target 10.0
```

Output:

```text
artifacts/coreml/compiled-watchos-prefill-kv-16-prefill-protected-no-int4/prefill-kv-16-mixed.mlmodelc/coremldata.bin
```

## Swift Diagnostic

Command:

```text
WATCHLM_RUN_REAL_COREML_TESTS=1 swift test --filter coreMLPrefillDecodeDiagnosticsCanRunLocalMiniCPMPrefillProtectedArtifacts
```

Relevant output:

```text
protected-no-int4-prefill-fp16-decode prefill=[416, 242, 1974, 359, 1030] decode=[826, 4245, 11420, 5018, 2793] decodeMargin=0.000000
protected-no-int4-prefill-int8-decode prefill=[416, 242, 1974, 359, 1030] decode=[4245, 826, 11420, 5018, 2793] decodeMargin=0.031250
```

The deployment candidate path is protected-no-int4 prefill plus int8 decode, and it restores the expected top-1 `4245`.

## Batch Benchmark

Command:

```text
swift run WatchLMBenchmark --runtime coreml --prefill artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected-no-int4/prefill-kv-16-mixed.mlpackage --decode artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json --teacher artifacts/benchmarks/minicpm5-teacher-references-context16-full.json --prompt-ids zh-short-001,en-short-001,code-fix-001,watch-utility-001,safety-refusal-001 --max-new-tokens 2 --context 16 --policy-id prefill-kv-fp16-attn-ffn-int8-decode-int8 --id real-minicpm5-context16-prefill-protected-no-int4-int8-decode-category-balanced --output artifacts/benchmarks/prefill-protected-no-int4-int8-decode-category-balanced.json
```

Summary:

```text
prompts: 5/5
average token agreement: 1.0
average first token: 2453.88 ms
average decode: 86.97 tokens/sec
peak resident memory: 2610.58 MB
load: 16661.957 ms
total artifact bytes: 2345277246
```

Prompt results:

```text
zh-short-001        [18487,45105] agreement 1.0
en-short-001        [416,4245]    agreement 1.0
code-fix-001        [3342,801]    agreement 1.0
watch-utility-001   [354,2305]    agreement 1.0
safety-refusal-001  [1974,220]    agreement 1.0
```

## Interpretation

The code prompt regression came from the layer-12 FFN int4 override. Removing that override costs about 10.6MB versus the protected layer-12-int4 artifact, but it restores 5/5 prompt agreement in the capped context-16 Swift benchmark.

This is now the best context-16 fidelity baseline for the next SE2/SE3-oriented experiments:

```text
prefill: attention/KV fp16, FFN int8, embedding/lm_head int8
decode: global int8
context: 16 smoke baseline
```

It is still not a deployable watch-size artifact. The next useful step is to test the same policy at a larger but still watch-relevant context, starting with context 256.
