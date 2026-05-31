# Qwen Watch SE Candidate Branch

## Purpose

Keep advanced non-MiniCPM model experiments isolated from the MiniCPM5-1B
foundation branch. The Qwen work now lives on:

```text
codex/qwen-watch-se-runtime
```

MiniCPM5-1B remains the teacher/reference line. This branch is for trying
newer small runtime candidates and deciding whether they can become practical
Apple Watch SE2/SE3 Core ML artifacts.

## Candidate Ranking

The first advanced model to try is:

```text
Qwen/Qwen3-0.6B
```

Reasons:

- Official model card reports 0.6B total parameters and 0.44B non-embedding
  parameters.
- It is a text-only causal LM, so it is closer to the current Swift/CoreML
  inference chain than Qwen3.5-0.8B.
- It uses 28 layers, 16 query heads, 8 KV heads, 128 head dimension, and a
  151936-token vocabulary.
- A context-256 int4 planning profile fits the current SE2 sizing gate.

The most advanced stretch candidate is:

```text
Qwen/Qwen3.5-0.8B
```

Reasons it is not first:

- It is a causal language model with a vision encoder in the released artifact.
- The text config uses a hybrid Qwen3.5 architecture with linear-attention
  layers and full-attention layers, which is not yet covered by the existing
  Llama/Qwen2-style stateful-step wrapper.
- Its padded vocabulary is 248320 tokens, which increases embedding/lm-head
  footprint even if word embeddings are tied.
- The SE2 planning gate currently rejects the conservative text-only int4
  profile due to estimated peak RSS.

SmolLM2-135M remains a lower-risk fallback because its Llama-like architecture
is closer to the current converter, but it is not the first experiment because
the user explicitly asked whether newer Qwen-class small models can run.

## Current Gate Output

Use:

```sh
node tools/validation/watchlm-validate.js candidates tools/validation/fixtures/model-candidates.json
```

Expected high-level result:

```text
candidates ok: 6 candidates, 3 passing SE2 gate
recommended next: qwen3-0.6b-int4 (Qwen/Qwen3-0.6B)
```

## Next Step

Attempt a real Qwen3-0.6B Core ML `stateful-step-kv` conversion at context 256.
If the current generic converter fails, keep the failure evidence in a separate
Qwen benchmark note and patch only the model-family-specific gap, rather than
weakening MiniCPM manifest or runtime contracts.

The Qwen-specific runtime contract is now tracked in
`docs/architecture/2026-05-31-qwen3-runtime-inference-contract.md`.
