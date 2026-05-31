# Qwen Explicit-KV Swift Smoke

Date: 2026-05-31

## Goal

Run a real Swift/CoreML inference path for `Qwen/Qwen3-0.6B` with explicit prefill/decode KV IO before resuming memory cleanup or deployment packaging.

## Runtime Contract Fix

The previous MiniCPM explicit-KV path assumed fixed graph dimensions:

- layers: 24
- KV heads: 2
- head dimension: 128

Qwen3-0.6B needs:

- layers: 28
- KV heads: 8
- head dimension: 128

The Swift runtime now builds `CoreMLPrefillDecodeBundle.layeredKV(...)` from CLI-provided graph dimensions instead of hard-coding the MiniCPM dimensions.

The conversion script also now preserves the dtype of decode `past_key_N` and `past_value_N` tensors. This matters for precision baselines: full-fp32 prefill emits fp32 KV, so decode must accept fp32 past KV rather than hard-coded fp16.

## Artifacts Used

- Prefill KV: `artifacts/coreml/qwen3-0.6b-prefill-kv-16-full-fp32/prefill-kv-16.mlpackage`
- Decode KV: `artifacts/coreml/qwen3-0.6b-decode-16-full-fp32/decode-16.mlpackage`
- Tokenizer: `artifacts/hf/Qwen3-0.6B/tokenizer.json`

These are full-fp32 context-16 artifacts. They are intentionally not watch deployment candidates; they are a correctness baseline for the Swift inference chain.

## Result

Report:

`artifacts/benchmarks/qwen3-0.6b-explicit-kv-16-full-fp32-swift-smoke.json`

Summary:

- Prompts: 1/1 succeeded
- Generated tokens: 4
- Output text: ` Please check the notification`
- First token / prefill: 164.832 ms
- Decode step ms: 162.411, 63.86, 58.956
- Average decode speed: 10.52 tokens/s
- Peak host RSS: 4667.03 MB
- Load time: 22116.832 ms
- KV strategy: slot ring

## Interpretation

This proves the Qwen branch now has a real Swift/CoreML path:

`Tokenizer -> PrefillGraph(logits + present KV) -> Swift KVStore -> DecodeGraph(logits + new KV) -> Sampler -> Streaming result`

It is still not deployable on Apple Watch SE 2/3 because the baseline pair is about 4.78 GB on disk and peaks around 4.67 GB RSS on host. The next inference-focused step is to move this same now-working graph contract to smaller Qwen artifacts: fp16/mixed precision first, then int4/stateful-step once quality and graph IO are stable.
