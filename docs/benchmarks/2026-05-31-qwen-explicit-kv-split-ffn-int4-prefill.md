# Qwen Explicit-KV Split-FFN Int4 Prefill Probes

Date: 2026-05-31

## Goal

Find a narrower Qwen3-0.6B int4 entry point after global int4 and broad FFN
int4 both caused unacceptable logits drift. These probes keep the known quality
bridge intact by leaving embedding, lm_head, attention, and non-selected FFN
weights at int8, then palettize only selected FFN subcomponents to uniform int4.

## Policies

```text
tools/conversion/mixed-precision-policy-qwen3-explicit-kv-ffn-down-low4-int4.json
tools/conversion/mixed-precision-policy-qwen3-explicit-kv-ffn-gateup-low4-int4.json
```

Selected layers:

```text
ffnDown int4:   6, 7, 9, 10
ffnGateUp int4: 5, 6, 7, 9
```

The layer choices are borrowed from the prior split-FFN importance experiment as
an initial heuristic. They are not yet Qwen-specific activation evidence.

## Artifacts

```text
artifacts/coreml/qwen3-0.6b-prefill-kv-16-qwen-ffn-down-low4-int4/prefill-kv-16-mixed.mlpackage
artifacts/coreml/qwen3-0.6b-prefill-kv-16-qwen-ffn-gateup-low4-int4/prefill-kv-16-mixed.mlpackage
```

Sizes:

```text
ffnDown low4:   591,993,386 bytes
ffnGateUp low4: 585,660,931 bytes
```

## Prefill Gate

Prompt:

```text
Apple Watch local inference test.
```

Gate:

```text
top-k: >= 9/10
top-1: required
mean absolute error: <= 0.5
```

Results:

```text
ffnDown low4:
  topKAgreement:    7/10
  top1Matches:      true
  meanAbsoluteError: 0.4040119052
  gate:             fail

ffnGateUp low4:
  topKAgreement:    7/10
  top1Matches:      true
  meanAbsoluteError: 0.6553533077
  gate:             fail
```

Reports:

```text
artifacts/benchmarks/qwen3-explicit-kv-ffn-down-low4-int4-prefill-validate.json
artifacts/benchmarks/qwen3-explicit-kv-ffn-gateup-low4-int4-prefill-validate.json
```

## Interpretation

Both probes are real Core ML prefill runs and both preserve the teacher top-1
token, but neither is clean enough to advance to decode and Swift runtime smoke.
`ffnGateUp` is more damaging than `ffnDown` on this prompt because it misses the
MAE gate as well as the top-k gate.

The useful signal is that Qwen3-0.6B cannot simply reuse the MiniCPM split-FFN
layer candidates. The next useful experiment should narrow further to
single-layer or Qwen-specific activation-selected modules before generating
decode artifacts.
