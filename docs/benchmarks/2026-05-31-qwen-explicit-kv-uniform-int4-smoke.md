# Qwen Explicit-KV Uniform Int4 Smoke

Date: 2026-05-31

## Goal

Take the working Qwen explicit-KV mixed baseline and test whether direct global int4 palettization can produce a smaller Swift/CoreML inference path.

This experiment keeps the proven split graph contract:

```text
prefill-kv -> Swift KVStore -> decode-kv
```

It changes only the storage compression.

## Conversion Route

The original global int4 path used CoreMLTools k-means palettization. Decode compression failed because the local environment does not have `scikit-learn`, which CoreMLTools needs for k-means on some weights.

The conversion CLI now exposes:

```text
--int4-mode kmeans|uniform
```

Using `--int4-mode uniform` avoids that dependency and completed both split graphs.

Artifacts:

- Prefill: `artifacts/coreml/qwen3-0.6b-prefill-kv-16-fp16-load-fp32-compute-int4/prefill-kv-16-int4.mlpackage`
- Decode: `artifacts/coreml/qwen3-0.6b-decode-16-fp16-load-fp32-compute-int4/decode-16-int4.mlpackage`

Sizes:

- Prefill: 299,148,020 bytes
- Decode: 299,250,965 bytes
- Tokenizer: 11,422,654 bytes
- Total benchmark payload: 609,821,639 bytes

## Runtime Fix

The first int4 smoke selected token `151680`, which is inside Qwen's Core ML logits dimension but outside the tokenizer's decodable token range.

Local tokenizer facts:

- Core ML logits: 151,936 tokens
- Qwen tokenizer decodable upper bound: 151,669
- Last decodable token ID: 151,668 (`</think>`)

The Swift runtime now exposes `TextTokenizer.decodableTokenIDUpperBound` and suppresses logits at or above that bound before sampling. This prevents reserved/undecodable token IDs from crashing streaming decode.

## Swift Smoke Result

Report:

`artifacts/benchmarks/qwen3-0.6b-explicit-kv-16-uniform-int4-swift-smoke.json`

Summary:

- Prompts: 1/1 succeeded
- Generated token IDs: `[27554, 27554, 27554, 27554]`
- Output text: `οοοο`
- First token / prefill: 81.818 ms
- Decode step ms: 64.575, 13.763, 13.37
- Average decode speed: 32.71 tok/s
- Peak host RSS: 2622.73 MB
- Load time: 9168.194 ms

## Teacher Drift

Prefill validation report:

`artifacts/benchmarks/qwen3-0.6b-prefill-kv-16-uniform-int4-validate.json`

Result:

- Top-k agreement: 0/10
- Top-1 match: false
- Mean absolute error: 4.1867
- Max absolute error: 21.8545

Teacher top-10:

```text
[1096, 576, 6771, 3555, 2585, 358, 220, 758, 481, 4710]
```

Core ML int4 top-10:

```text
[27781, 10864, 51131, 25354, 92002, 40376, 25232, 130079, 70019, 76858]
```

## Interpretation

This is useful infrastructure progress but not a deployable model candidate.

What improved:

- The explicit-KV int4 split graph runs through real Swift inference.
- The artifact payload is now around 610 MB instead of 2.52 GB mixed or 4.78 GB full-fp32.
- Reserved tokenizer IDs are now suppressed during sampling.

What failed:

- Global uniform int4 destroys Qwen prefill logits.
- Output collapses to repeated token `ο`.
- Host RSS remains around 2.62 GB because split prefill/decode duplicates weights.

Next direction:

- Do not promote global uniform int4.
- Keep explicit-KV as the correctness/debug path.
- Use mixed/layer-aware int4 policies for quality.
- Use a shared stateful-step or other single-artifact route for actual SE2/SE3 memory deployment once fp16/mixed correctness is protected.
