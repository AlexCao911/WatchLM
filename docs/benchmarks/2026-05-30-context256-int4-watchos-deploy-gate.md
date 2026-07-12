# Context-256 Int4 watchOS Deploy Gate

Date: 2026-05-30

## Scope

This note records the first context-256 deployment-first Core ML candidate after moving GGUF work off the main branch.

The goal of this run was not quality promotion. It was to answer whether a context-256 MiniCPM5 prefill/decode pair can be made small enough to pass watchOS Core ML compilation for the Watch SE2 target path.

## Why the Previous Pair Was Large

The previous context-256 pair used two separate full-model Core ML graphs:

- prefill graph: `artifacts/coreml/real-minicpm5-prefill-kv-256-prefill-protected-no-int4/prefill-kv-256-mixed.mlpackage`
- decode graph: `artifacts/coreml/real-minicpm5-decode-256-int8/decode-256-int8.mlpackage`

That duplicates MiniCPM5 weights across prefill and decode. The protected prefill policy also keeps attention, KV-producing paths, and norms at fp16 while using int8 for embedding, lm_head, and FFN. The result is about 2.35 GB before packaging overhead.

## Int4 Deployment Candidate

Generated from existing fp16 context-256 packages:

```text
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py --graph decode --source-mlpackage artifacts/coreml/real-minicpm5-decode-256/decode-256.mlpackage --compression int4 --output-dir artifacts/coreml/real-minicpm5-decode-256-int4

.venv/bin/python tools/conversion/convert-minicpm5-coreml.py --graph prefill-kv --context-tokens 256 --source-mlpackage artifacts/coreml/real-minicpm5-prefill-kv-256/prefill-kv-256.mlpackage --compression int4 --output-dir artifacts/coreml/real-minicpm5-prefill-kv-256-int4
```

Artifact sizes:

```text
prefill-kv-256-int4.mlpackage: 541,184,228 bytes
decode-256-int4.mlpackage:     541,248,864 bytes
tokenizer.json:                  9,894,271 bytes
total benchmark artifact bytes: 1,092,327,363 bytes
```

This cuts the previous 2.35 GB pair to about 1.09 GB, but it still carries two full copies of the model weights.

## watchOS Compile

Commands:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile artifacts/coreml/real-minicpm5-prefill-kv-256-int4/prefill-kv-256-int4.mlpackage artifacts/coreml/compiled-watchos-prefill-kv-256-int4 --platform watchOS --deployment-target 10.0

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile artifacts/coreml/real-minicpm5-decode-256-int4/decode-256-int4.mlpackage artifacts/coreml/compiled-watchos-decode-256-int4 --platform watchOS --deployment-target 10.0
```

Both commands succeeded and produced:

```text
artifacts/coreml/compiled-watchos-prefill-kv-256-int4/prefill-kv-256-int4.mlmodelc
artifacts/coreml/compiled-watchos-decode-256-int4/decode-256-int4.mlmodelc
```

Compiled sizes:

```text
compiled prefill: 516 MB
compiled decode:  516 MB
```

## Swift Inference Smoke

Command:

```text
swift run WatchLMBenchmark --runtime coreml --prefill artifacts/coreml/real-minicpm5-prefill-kv-256-int4/prefill-kv-256-int4.mlpackage --decode artifacts/coreml/real-minicpm5-decode-256-int4/decode-256-int4.mlpackage --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json --teacher artifacts/benchmarks/minicpm5-teacher-references-context256-cap2.json --prompt-ids en-short-001 --max-new-tokens 2 --context 256 --policy-id global-int4-deploy-first --id real-minicpm5-context256-int4-deploy-first-smoke --output artifacts/benchmarks/context256-int4-deploy-first-smoke.json
```

Result:

```text
prompts: 1/1
average token agreement: 0.0
generated token IDs: [39, 34]
average first token: 263.66 ms
average decode throughput: 1.49 tokens/sec
peak resident memory on host: 3627.47 MB
```

## Interpretation

The context-256 int4 pair passes the watchOS Core ML compile gate, so it is a real deployment artifact candidate in the packaging sense.

It is not yet a physical Watch SE2 runtime pass:

- quality is not acceptable under global int4
- host load memory increased, which suggests Core ML palettized int4 may carry load-time expansion or compilation overhead
- the split prefill/decode architecture still duplicates full model weights

The next SE2-focused work should therefore target one of these routes:

1. Reduce duplicate weights by moving toward a single stateful Core ML program or a decode-only prefill fallback.
2. Keep the deployable int4 size but recover quality with selective fp16/int8 protection around prefill attention/KV paths.
3. Add a physical-device memory gate before promoting any artifact beyond "watchOS compile passed".

