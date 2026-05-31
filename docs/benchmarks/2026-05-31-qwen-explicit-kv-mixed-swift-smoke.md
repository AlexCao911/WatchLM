# Qwen Explicit-KV Mixed Swift Smoke

Date: 2026-05-31

## Goal

Move the working Qwen explicit-KV Swift/CoreML path from the full-fp32 correctness baseline to a smaller mixed-precision baseline:

```text
fp16 model load + Core ML fp32 compute
```

This is still not the Apple Watch deployment target. It is the next inference-chain baseline before int4/stateful-step optimization.

## Runtime Fix

The mixed baseline uses fp16 KV from prefill and fp32 `new_key_N` / `new_value_N` from decode:

```text
prefill present KV: fp16
decode past KV input: fp16
decode new KV output: fp32
Swift KV store: fp16
```

`CoreMLKVCacheStore` now keeps strict shape validation but allows fp16/fp32 decode slices to be appended into a fp16/fp32 KV store. This lets the runtime preserve a compact KV cache while accepting fp32 compute outputs from Core ML.

## Artifacts Used

- Prefill KV: `artifacts/coreml/qwen3-0.6b-prefill-kv-16-gate-fp32/prefill-kv-16.mlpackage`
- Decode KV: `artifacts/coreml/qwen3-0.6b-decode-16-gate-fp32/decode-16.mlpackage`
- Tokenizer: `artifacts/hf/Qwen3-0.6B/tokenizer.json`

Artifact sizes:

- Prefill: 1,193,069,578 bytes
- Decode: 1,310,613,042 bytes
- Tokenizer: 11,422,654 bytes
- Total: 2,515,105,274 bytes

## Result

Report:

`artifacts/benchmarks/qwen3-0.6b-explicit-kv-16-fp16-load-fp32-compute-swift-smoke.json`

Summary:

- Prompts: 1/1 succeeded
- Generated tokens: 4
- Output text: ` Please check the notification`
- Generated token IDs: `[5209, 1779, 279, 11540]`
- First token / prefill: 81.741 ms
- Decode step ms: 65.111, 11.11, 14.143
- Average decode speed: 33.2 tokens/s
- Peak host RSS: 2485.53 MB
- Load time: 8460.328 ms
- KV strategy: slot ring

## Comparison With Full-fp32 Baseline

The mixed baseline produced the same four smoke tokens as the full-fp32 baseline:

```text
[5209, 1779, 279, 11540]
```

Compared with the full-fp32 explicit-KV smoke:

- Total artifact size: 4.78 GB -> 2.52 GB
- Peak host RSS: 4667.03 MB -> 2485.53 MB
- First token: 164.832 ms -> 81.741 ms
- Decode speed: 10.52 tok/s -> 33.2 tok/s
- Load time: 22116.832 ms -> 8460.328 ms

## Interpretation

This proves a useful mixed-precision Qwen inference route:

```text
Tokenizer -> fp16 prefill KV -> Swift fp16 KVStore -> fp32-compute decode -> fp16 KV append -> sampler
```

The path is still much too large for Apple Watch SE2/SE3, but it gives a better baseline than full-fp32 for the next optimization step. The next target should keep this explicit-KV correctness path intact while reducing weights with int4 or moving back to stateful-step only after fp16/mixed logits are known to align.
