# Qwen Stateful-Step Narrow FFN Down Int4

Date: 2026-05-31

## Goal

After global int4 and broad FFN int4 failed, test a much narrower Qwen
stateful-step compression candidate:

```text
only FFN down projection in layers 6, 7, 9, and 10 uses int4
everything else remains int8/fp16 according to the policy
```

The source graph is the validated fp32-compute stateful-step graph:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256.mlpackage
```

## Policy

```text
tools/conversion/mixed-precision-policy-qwen3-explicit-kv-ffn-down-low4-int4.json
```

Compression audit:

```text
int8 selected ops: 193
int4 selected ops: 4
int4 layers: 6, 7, 9, 10
int4 component: ffnDown
```

## Artifact

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-mixed-ffn-down-low4-int4/stateful-step-kv-256-mixed.mlpackage
```

Size:

```text
mlpackage: 592,134,628 bytes
tokenizer:  11,422,654 bytes
total:     603,557,282 bytes
du size:   565 MB
```

This is only slightly smaller than the current int8 candidate:

```text
int8 total: 609,855,974 bytes
```

## Swift Smoke

Report:

```text
artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-fp32-compute-mixed-ffn-down-low4-int4-swift-smoke.json
```

Observed:

```text
generatedTokenIDs: [785, 1614, 9329, 702]
text: "The model asset has"
firstTokenMs: 570.737
averageDecodeTokensPerSecond: 62.69
peakResidentMemoryMB: 1595.44
```

The first three generated tokens match the int8 baseline:

```text
int8 baseline: [785, 1614, 9329, 374]
candidate:    [785, 1614, 9329, 702]
```

## Top-K Diagnostics

Report:

```text
artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-fp32-compute-mixed-ffn-down-low4-int4-topk.json
```

Full-prefix top-k:

```text
prefill top1: token 785
decode top1:  token 1614
```

This means the candidate preserves the first-token and first-decode gate. The
drift appears later in the decode loop, where the fourth generated token flips
from `374` to `702`.

## Interpretation

This candidate is useful as a direction signal, but not as a deployment
candidate:

```text
quality: closer than global/broad int4, but still drifts by token 4
size: only about 6.3 MB smaller than int8 total payload
memory: host RSS is roughly unchanged
```

The next useful compression search should either:

1. Use a wider but still measured narrow set that actually reduces size, while
   checking multi-token decode agreement, or
2. Add a multi-step top-k/teacher gate so candidates that pass first decode but
   drift by token 4 are rejected automatically.
