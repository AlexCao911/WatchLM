# Qwen3 Runtime Inference Contract

## Purpose

Define the Qwen-specific inference contract before continuing Apple Watch SE2/SE3
runtime work. The existing Swift/CoreML infrastructure was first proven against
MiniCPM-shaped artifacts, so Qwen must be treated as a separate model family
configuration rather than a MiniCPM artifact with different dimensions.

## Local Config Snapshot

Source:

```text
artifacts/hf/Qwen3-0.6B/config.json
```

Relevant values:

```text
architecture: Qwen3ForCausalLM
model_type: qwen3
layers: 28
hidden_size: 1024
intermediate_size: 3072
query_heads: 16
kv_heads: 8
head_dim: 128
vocab_size: 151936
tie_word_embeddings: true
rope_theta: 1000000
rope_scaling: null
sliding_window: null
use_sliding_window: false
source dtype: bfloat16
bos_token_id: 151643
eos_token_id: 151645
```

## Required Qwen Adapter Surface

The runtime path needs a Qwen adapter with these explicit choices:

```text
tokenizer.add_bos_token: false
tokenizer.eos_token_ids: [151645]
chat_template: qwen3-nonthinking
graph.layer_count: 28
graph.kv_heads: 8
graph.head_dim: 128
sampler.input: logits, not next_token
context_variant: 256 first, then 512 only after correctness
```

These values should come from a manifest/runtime candidate configuration, not
from MiniCPM defaults inside command-line invocations.

## What Can Stay Generic

The following infrastructure should remain shared:

```text
Tokenizer -> ChatTemplate -> GraphRunner -> LogitsProcessor -> Sampler
KV layout validation
Core ML model loading and compute-unit selection
benchmark metrics and prompt selection
teacher-vs-runtime top-k reports
watch SE profile gates
```

The generic path should consume a model-family adapter instead of hard-coding
MiniCPM assumptions.

## What Must Be Rechecked for Qwen

The current blocker is not simply a missing dimension override. The Qwen int4
and fp16 Core ML stateful-step graphs both drift from PyTorch teacher logits.
The following semantics must be verified against Qwen specifically:

1. Stateful KV state shape and layer order:
   `layers x {key,value} x [1, kv_heads, context, head_dim]`.
2. Position IDs during prompt scan and decode:
   monotonic positions must match Qwen's RoPE expectations.
3. Causal mask shape and blocked values:
   stateful-step uses `[1, 1, 1, context + 1]`; its allowed slots must match the
   actual state update strategy.
4. KV update policy:
   the current sliding `cat(old[:, :, 1:, :], new)` style may not lower to Core
   ML state updates in a way that preserves Qwen teacher semantics.
5. Output contract:
   Core ML must output logits `[1, vocab_size]`; Swift sampler selects from
   logits and should never depend on a graph-produced `next_token`.

## Next Architecture Step

Before more quantization work, add a correctness gate:

```text
PyTorch teacher top-k
vs
Core ML fp16 graph top-k
```

Only after fp16 top-k agreement is acceptable should the branch resume int4
palettization, mixed precision policy experiments, watchOS compile, and
SE2/SE3 latency/memory tuning.

The next graph experiment should be an explicit-KV or non-stateful Qwen sanity
graph. If that aligns with the teacher, the fault is isolated to the Core ML
stateful update pattern. If it does not align, the conversion wrapper or input
contract needs to be corrected before stateful KV is revisited.

## Precision Update

The first non-stateful Qwen sanity graph isolated the immediate failure to Core
ML `compute_precision=float16`. A logits-only prefill graph matches teacher
top-k with `compute_precision=float32`, including when the Hugging Face model is
loaded as float16 before conversion.

This means Qwen needs a model-family-specific mixed compute precision route
before stateful KV and int4 storage compression can be trusted.

## Manifest Update

The Qwen explicit-KV runtime candidate now has a Swift-tested manifest fixture:

```text
tools/validation/fixtures/qwen3-0.6b-explicit-kv-model-manifest.json
```

The fixture records the Qwen-specific graph contract directly:

```text
interface: logits-layered-kv
layers: 28
kv_heads: 8
head_dim: 128
chat_template: qwen3-nonthinking
add_bos_token: false
bos_token_id: 151643
eos_token_ids: [151645]
watch-se-2 default context: 256
watch-se-3 default context: 512
```

This closes the architecture gap where Qwen could be run only by manually
passing `--coreml-layer-count 28 --coreml-kv-heads 8 --coreml-head-dim 128` to
the benchmark command. The formal manifest path can now feed
`CoreMLPrefillDecodeBundle(graphSchema:)` with Qwen dimensions instead of
MiniCPM defaults.

The Swift assembler and benchmark command now consume the tokenizer settings
from the manifest. `WatchLMBenchmark` can resolve the Qwen explicit-KV runtime
candidate with:

```text
swift run WatchLMBenchmark \
  --manifest tools/validation/fixtures/qwen3-0.6b-explicit-kv-model-manifest.json \
  --asset-base <artifact-root> \
  --device-profile watch-se-2
```

That path sets the Core ML graph dimensions, selected context variant, artifact
paths, tokenizer path, BOS/EOS policy, and chat template from one source of
truth before running the existing Swift prefill/decode/KV/sampler chain.
