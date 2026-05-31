# Qwen Explicit-KV Protected FFN Int4 Prefill

Date: 2026-05-31

## Goal

Test a layer-aware int4 step after the successful explicit-KV int8 bridge.

The policy is intentionally conservative:

- Qwen layer count: 28
- Edge protection: first 4 and last 4 layers
- Embedding / attention / lm head: int8
- Norms: fp16
- Middle-layer FFN: int4
- KV cache: fp16
- Int4 mode: uniform palettization

Policy:

`tools/conversion/mixed-precision-policy-qwen3-explicit-kv-ffn-int4-protected.json`

## Conversion

Source:

`artifacts/coreml/qwen3-0.6b-prefill-kv-16-gate-fp32/prefill-kv-16.mlpackage`

Output:

`artifacts/coreml/qwen3-0.6b-prefill-kv-16-qwen-ffn-int4-protected/prefill-kv-16-mixed.mlpackage`

Size:

```text
503,634,006 bytes
```

Compression audit:

- Int8 selected ops: 137
- Int4 selected ops: 60
- Int4 component: FFN only
- Int4 layers: 4-23
- Protected FFN layers promoted to int8: 0-3 and 24-27

## Teacher Gate

Report:

`artifacts/benchmarks/qwen3-0.6b-prefill-kv-16-qwen-ffn-int4-protected-validate.json`

Result:

- Top-k agreement: 2/10
- Top-1 match: false
- Mean absolute error: 2.9325
- Max absolute error: 15.2930

Teacher top-10:

```text
[1096, 576, 6771, 3555, 2585, 358, 220, 758, 481, 4710]
```

Core ML mixed top-10:

```text
[220, 0, 1154, 4894, 659, 11, 262, 753, 481, 256]
```

## Interpretation

This policy is not viable. It improves size relative to int8 prefill:

```text
598,292,437 bytes -> 503,634,006 bytes
```

But the quality loss is too large. Compared with the int8 bridge:

```text
int8 top-k agreement:          9/10
protected FFN int4 agreement:  2/10
```

The useful signal is that Qwen3-0.6B is very sensitive to broad FFN int4. The next int4 attempt should not quantize the whole FFN block. It should split the FFN path and test narrower policies:

- `ffnDown` only
- `ffnGateUp` only
- fewer middle layers

This keeps the search guided by the failure mode instead of widening int4 blindly.
