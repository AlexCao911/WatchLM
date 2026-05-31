# Qwen Explicit-KV Int8 Smoke

Date: 2026-05-31

## Goal

Test a quality-preserving compression step between the mixed fp16/fp32 baseline and the failed global uniform int4 baseline.

This answers a narrow question:

```text
Does explicit-KV compression itself break Qwen, or is the quality collapse mainly from global int4?
```

## Artifacts

Source graphs:

- `artifacts/coreml/qwen3-0.6b-prefill-kv-16-gate-fp32/prefill-kv-16.mlpackage`
- `artifacts/coreml/qwen3-0.6b-decode-16-gate-fp32/decode-16.mlpackage`

Compressed int8 graphs:

- `artifacts/coreml/qwen3-0.6b-prefill-kv-16-fp16-load-fp32-compute-int8/prefill-kv-16-int8.mlpackage`
- `artifacts/coreml/qwen3-0.6b-decode-16-fp16-load-fp32-compute-int8/decode-16-int8.mlpackage`

Sizes:

- Prefill: 598,292,437 bytes
- Decode: 598,452,728 bytes
- Tokenizer: 11,422,654 bytes
- Total benchmark payload: 1,208,167,819 bytes

## Teacher Gate

Report:

`artifacts/benchmarks/qwen3-0.6b-prefill-kv-16-int8-validate.json`

Result:

- Top-k agreement: 9/10
- Top-1 match: true
- Mean absolute error: 0.0707
- Max absolute error: 0.4248

Strict 10/10 top-k gate failed by one token, but this is dramatically better than global uniform int4, which had 0/10 top-k agreement and repeated-token output.

Teacher top-10:

```text
[1096, 576, 6771, 3555, 2585, 358, 220, 758, 481, 4710]
```

Core ML int8 top-10:

```text
[1096, 2585, 576, 3555, 6771, 358, 220, 481, 758, 362]
```

## Swift Smoke

Report:

`artifacts/benchmarks/qwen3-0.6b-explicit-kv-16-int8-swift-smoke.json`

Summary:

- Prompts: 1/1 succeeded
- Generated token IDs: `[5209, 1779, 279, 11540]`
- Output text: ` Please check the notification`
- First token / prefill: 80.494 ms
- Decode step ms: 67.699, 11.458, 11.05
- Average decode speed: 33.26 tok/s
- Peak host RSS: 2668.02 MB
- Load time: 6487.184 ms

## Interpretation

Int8 preserves the smoke output from the mixed and full-fp32 explicit-KV baselines:

```text
[5209, 1779, 279, 11540]
```

This makes int8 a useful fidelity bridge for Qwen. It suggests the global uniform int4 failure is mostly an overly aggressive 4-bit compression problem, not a general failure of the explicit-KV split graph.

It is still not a deployment candidate for Apple Watch SE2/SE3:

- The split graph payload is still about 1.21 GB.
- Host RSS is still about 2.67 GB because prefill and decode duplicate weights.

Next direction:

- Use int8 as the quality-preserving reference for layer-aware int4 experiments.
- Keep explicit-KV as the correctness/debug path.
- For SE2/SE3 deployment, move the protected precision policy back into a single shared/stateful graph after the Qwen fp16/mixed semantics are trusted.

## Rerun Note

After adding the Qwen manifest contract, the same int8 artifact was rerun to
check the Swift inference chain. The historical output is reproduced when the
benchmark uses the raw prompt format:

```text
report: artifacts/benchmarks/qwen3-0.6b-explicit-kv-16-int8-swift-smoke-rerun-raw.json
generatedTokenIDs: [5209, 1779, 279, 11540]
text: " Please check the notification"
firstTokenMs: 72.661
averageDecodeTokensPerSecond: 35.68
peakResidentMemoryMB: 2650.23
```

The same context16 artifact with `qwen3-nonthinking` chat template generated:

```text
report: artifacts/benchmarks/qwen3-0.6b-explicit-kv-16-int8-swift-smoke-rerun-all.json
generatedTokenIDs: [9454, 11, 323, 323]
text: "Yes, and and"
```

Interpretation: context16 is valid as a small graph/runtime smoke, but it is too
short for a real Qwen chat-template prompt. Qwen user-facing inference and
quality measurements should use the context256 path, while context16 remains a
debug artifact for graph IO, KV cache append, logits sampling, and load
behavior.
