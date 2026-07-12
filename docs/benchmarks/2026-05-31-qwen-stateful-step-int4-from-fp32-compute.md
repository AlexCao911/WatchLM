# Qwen Stateful-Step Int4 From FP32-Compute Source

Date: 2026-05-31

## Goal

Test whether the Qwen stateful-step graph can move from the validated
fp32-compute int8 candidate toward int4-sized artifacts without losing the
Swift inference top-k gate.

The source graph for every experiment here is:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256.mlpackage
```

This matters because earlier Qwen stateful int4 artifacts were built from a
float16-compute source graph that already had incorrect logits.

## Baseline To Preserve

Validated candidate:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlpackage
```

Swift output on `watch-utility-002`:

```text
generatedTokenIDs: [785, 1614, 9329, 374]
text: "The model asset is"
```

Full-prefix top-k:

```text
prefill top1: token 785
decode top1:  token 1614
```

## Global Int4 K-Means

Attempted:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --model-id Qwen/Qwen3-0.6B \
  --source-mlpackage artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256.mlpackage \
  --graph stateful-step-kv \
  --context-tokens 256 \
  --output-dir artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int4 \
  --compression int4 \
  --int4-mode kmeans
```

Result:

```text
status: failed
failure: scikit-learn is required for k-means quantization
```

This is an environment dependency blocker, not a model quality result.

## Global Int4 Uniform

Artifact:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int4-uniform/stateful-step-kv-256-int4.mlpackage
```

Size:

```text
mlpackage: 299,405,263 bytes
tokenizer:  11,422,654 bytes
total:     310,827,917 bytes
du size:   286 MB
```

Swift smoke:

```text
generatedTokenIDs: [27554, 94108, 15337, 1003]
text: "ο Heardalandrit"
firstTokenMs: 560.849
averageDecodeTokensPerSecond: 62.89
peakResidentMemoryMB: 1592.2
```

Full-prefix top-k:

```text
prefill top1: token 151680
decode top1:  token 94108
```

The artifact is deployably small, but quality fails. The top-k failure pushes
Qwen into high special-token IDs, which is a strong signal that global int4
damages sensitive output-distribution weights.

## Mixed FFN Int4 Protected

Policy:

```text
tools/conversion/mixed-precision-policy-qwen3-explicit-kv-ffn-int4-protected.json
```

Policy shape:

```text
embedding: int8
lmHead: int8
norms: fp16
attention Q/K/O: int8
attention V: int8
middle-layer FFN: int4
edge-layer FFN: int8
KV cache: fp16
```

Artifact:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-mixed-ffn-int4-protected/stateful-step-kv-256-mixed.mlpackage
```

Size:

```text
mlpackage: 503,779,604 bytes
tokenizer:  11,422,654 bytes
total:     515,202,258 bytes
du size:   480 MB
```

Swift smoke:

```text
generatedTokenIDs: [315, 6321, 11, 3638]
text: " ofsi, diff"
firstTokenMs: 562.863
averageDecodeTokensPerSecond: 63.34
peakResidentMemoryMB: 1550.25
```

Full-prefix top-k:

```text
prefill top1: token 315
decode top1:  token 6321
```

The protected policy reduces size versus int8, but still fails the Qwen
stateful top-k gate. Middle-layer FFN int4 is too broad for this model/runtime
combination.

## Interpretation

The useful new fact is not "int4 cannot work"; it is narrower:

```text
Qwen stateful-step can be correct from a float32-compute source,
but global uniform int4 and broad middle-layer FFN int4 both break top-k.
```

The next int4 search should be more surgical:

1. Keep the fp32-compute source graph as the only source of truth.
2. Keep embedding, lm_head, norms, and attention at least int8/fp16.
3. Try very narrow FFN down-proj or gate/up layer subsets first.
4. Promote only candidates that keep full-prefix prefill top1 `785` and decode
   top1 `1614` on the Qwen watch prompt before expanding to more prompts.
