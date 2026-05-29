# Int4 Decode KV Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add the first real MiniCPM5 int4 compression path and generate Core ML prefill/decode graphs with explicit KV cache IO.

**Architecture:** Extend the existing conversion spike instead of creating a second tool. The script can now choose graph shape (`prefill`, `prefill-kv`, `decode`) and compression (`none`, `int8`, `int4`). KV cache construction stays outside the watch runtime for this spike, while Core ML graph IO is named exactly as the Swift runtime should consume later.

**Tech Stack:** Python, PyTorch, Transformers DynamicCache, coremltools ML Program conversion, Core ML palettization, Xcode `coremlc`.

---

## Task 1: Conversion CLI Surface

- [x] Add a failing Node test requiring `--compression none|int8|int4` and `--graph prefill|prefill-kv|decode`.
- [x] Add graph/compression arguments to `tools/conversion/convert-minicpm5-coreml.py`.
- [x] Preserve `--quantize` as a deprecated alias for `--compression int8`.
- [x] Run `node --test test/realConversionCli.test.js`.

## Task 2: Full-Model Int4 Spike

- [x] Implement Core ML int4 palettization through `OpPalettizerConfig(mode="kmeans", nbits=4)`.
- [x] Generate `artifacts/coreml/real-minicpm5-prefill-16-int4/prefill-16-int4.mlpackage`.
- [x] Compile it for watchOS 10 with `coremlc`.
- [x] Run PyTorch teacher vs Core ML logits validation.

Observed:

```text
int4 package bytes: 541164434
compiled size: about 516MB
top-1 match: false
top-10 agreement: 1/10
max absolute error: 15.26953125
```

Conclusion: full-model int4 is a valid size/compile proof, but it is not fidelity-safe. The next optimization pass should use mixed precision, not blanket int4.

## Task 3: Prefill KV Graph

- [x] Add `MiniCPMPrefillKVWrapper`.
- [x] Return `logits` plus `present_key_N` and `present_value_N` for all 24 layers.
- [x] Generate `artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage`.
- [x] Compile it for watchOS 10 with `coremlc`.

Observed:

```text
present_key/value shape: [1, 2, 16, 128]
package size: about 2.0GB
watchOS compile: succeeded
```

## Task 4: Decode KV Graph

- [x] Add `MiniCPMDecodeKVWrapper`.
- [x] Accept `token_id`, `position_id`, a decode causal mask, and `past_key/value_N` for all 24 layers.
- [x] Return `logits` plus one-token `new_key_N` and `new_value_N` for all 24 layers.
- [x] Generate `artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage`.
- [x] Compile it for watchOS 10 with `coremlc`.
- [x] Add `tools/validation/validate-coreml-decode.py`.
- [x] Run PyTorch teacher vs Core ML decode validation.

Observed:

```text
past_key/value shape: [1, 2, 16, 128]
new_key/value shape: [1, 2, 1, 128]
top-1 match: true
top-10 agreement: 9/10
logits max absolute error: 0.19140625
KV max absolute error: 0.087890625
```

## Next Work

- Replace full-model int4 with mixed compression: keep embedding/lm_head and sensitive attention paths at int8 or fp16, then palettize selected FFN/linear weights.
- Convert `prefill-kv` and `decode` with shared compression policy.
- Update Swift `CoreMLPrefillDecodeRuntime` to consume logits, run sampler policy, and append `new_key/value` into a preallocated KV cache instead of expecting a scalar `next_token` output.
- Move from `context=16` to SE2 `context=256` and SE3 `context=512`.
