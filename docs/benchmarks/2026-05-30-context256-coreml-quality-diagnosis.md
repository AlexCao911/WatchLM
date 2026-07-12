# Context-256 Core ML quality diagnosis

Date: 2026-05-30

## Scope

This note records the current context-256 Core ML inference evidence. It is separate from architecture notes and GGUF route assessment so the benchmark trail stays readable.

## Artifacts

- Prefill base fp16: `artifacts/coreml/real-minicpm5-prefill-kv-256/prefill-kv-256.mlpackage`
  - Size: 2,161,972,701 bytes
- Prefill mixed policy: `artifacts/coreml/real-minicpm5-prefill-kv-256-prefill-protected-no-int4/prefill-kv-256-mixed.mlpackage`
  - Size: 1,252,482,006 bytes
  - Policy: attention Q/K/O/V fp16, norms fp16, KV fp16, embedding/lm_head/FFN int8
  - Audit: 74 int8-selected ops, 0 int4-selected ops
- Decode fp16: `artifacts/coreml/real-minicpm5-decode-256/decode-256.mlpackage`
  - Size: 2,162,037,340 bytes
- Decode int8: `artifacts/coreml/real-minicpm5-decode-256-int8/decode-256-int8.mlpackage`
  - Size: 1,082,903,364 bytes

The deployable pair currently being tested is about 2.35 GB before app packaging overhead:

- prefill mixed: 1.25 GB
- decode int8: 1.08 GB

This is still too large for a credible Apple Watch SE2/SE3 product target.

## Findings

The Swift/Core ML chain is no longer a mock chain. It runs:

Tokenizer -> prefill Core ML graph -> logits sampler -> KV cache store -> decode Core ML graph -> logits sampler -> benchmark quality comparison.

The context-256 pair runs mechanically, but quality is not acceptable yet.

End-to-end context-256 benchmark:

- Report: `artifacts/benchmarks/prefill-protected-no-int4-int8-decode-category-balanced-context256-safe-mask.json`
- Prompts: 5/5 succeeded
- Average token agreement: 0.0
- Average first token latency on host: 2522.92 ms
- Average decode throughput on host: 32.46 tokens/sec
- Peak resident memory on host: 1639.17 MB

Standalone decode-256 validation is good:

- Report: `artifacts/benchmarks/validation-context256-int8-decode-en-short-001.json`
- logits top-10 agreement: 10/10
- top-1 matched: true
- logits mean absolute error: 0.150469

Prefill-256 remains the failing section:

- Before the safe padding mask change, CPU-only validation produced NaN logits.
- After the safe padding mask change, NaNs are removed for the same existing mlpackage inputs.
- With Core ML `ALL` compute units, context-256 mixed prefill still has top-10 agreement 0/10 and top-1 mismatch against PyTorch.

Latest matched validation:

- Report: `artifacts/benchmarks/validation-context256-protected-prefill-kv-safe-mask-all-graphmatched-en-short-001.json`
- graph: prefill-kv
- compute units: all
- top-10 agreement: 0/10
- top-1 matched: false
- max absolute error: 19.953125
- mean absolute error: 2.211821

## Code changes made in this chunk

- Added a safe pad-query self-attention path in Swift input-state mask construction.
- Mirrored that mask behavior in the Python conversion helper.
- Extended `validate-coreml-prefill.py` with:
  - `--graph prefill|prefill-kv`
  - `--compute-units cpu|all`

The validator change matters because context-16 prefill-kv appears bad under CPU-only Core ML but matches PyTorch under `ALL` compute units:

- context-16 mixed prefill-kv, `ALL`: top-10 agreement 10/10, top-1 matched true
- context-16 mixed prefill-kv, CPU-only: top-10 agreement 1/10, top-1 mismatch

## Current interpretation

The blocker is not decode/KV infrastructure. Decode-256 independently matches PyTorch when fed PyTorch KV.

The blocker is context-256 prefill quality and artifact size:

- quality: prefill logits diverge before decode gets a chance to be correct
- size: the current Core ML pair is about 2.35 GB

The next useful work should focus on prefill-256 graph structure and route choice, not more sampler work.

