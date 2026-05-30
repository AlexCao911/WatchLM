# Community Quantization Priors

Date: 2026-05-30

## Scope

This note records external quantization priors that should guide WatchLM's
MiniCPM5-1B Core ML search.

It is separate from per-experiment benchmark notes. Its purpose is to keep the
search hypothesis-driven instead of trying arbitrary layer/component
combinations.

## Sources Consulted

DS4:

```text
https://github.com/antirez/ds4
https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/dataset/README.md
```

llama.cpp / GGUF:

```text
https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
```

AWQ / PTQ:

```text
https://huggingface.co/papers/2306.00978
https://github.com/mit-han-lab/llm-awq
https://arxiv.org/abs/2210.17323
https://arxiv.org/abs/2211.10438
```

Core ML compression:

```text
https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html
https://apple.github.io/coremltools/docs-guides/source/opt-opt1_3.html
```

Community discussions used as lower-confidence priors:

```text
https://www.reddit.com/r/LocalLLaMA/comments/1fnctht/llamacpp_quantize_results_in_garbage_output_how/
https://www.reddit.com/r/LocalLLaMA/comments/1reqdpb/overwhelmed_by_so_many_quantization_variants/
```

## DS4 Prior

DS4 is not doing a blind sweep. It uses a model-specific asymmetric recipe:

```text
DeepSeek V4 routed MoE experts:
  gate/up experts: IQ2_XXS
  down experts:    Q2_K

Everything else:
  attention projections: Q8_0
  shared experts:        Q8_0
  output head:           Q8_0
  embedding:             F16
  routers / norms / bias F16 or F32
```

The important lesson is not "use 2-bit". The lesson is:

```text
Quantize the component class that dominates model size and is structurally less
central to routing / projection / output decisions. Keep decision-making and
distribution-shaping tensors higher precision.
```

For DS4, routed experts are a natural target because they dominate size and each
expert handles only part of the token stream. MiniCPM5-1B is a dense model, so
there is no equivalent routed-expert bulk target.

DS4 also moved from legacy non-imatrix downloads toward imatrix variants. Its
imatrix pipeline collects activation statistics with the runtime itself over a
rendered prompt corpus, then uses those statistics to guide the low-bit routed
expert quantization. The tracked dataset covers code review, long-context,
tool-call, multilingual, summarization, extraction, reasoning, and debugging
prompts.

The WatchLM takeaway is:

```text
Before another large low-bit sweep, build a representative WatchLM calibration
set and use runtime logits / activation evidence to rank tensor sensitivity.
```

## llama.cpp / GGUF Prior

The llama.cpp ecosystem has converged on mixed recipes rather than uniform
naive low-bit quantization:

```text
Q4_K_M / K-quants:
  mixed precision by tensor class and block layout, common practical baseline

IQ4_XS / newer i-quants:
  often used when memory is tighter, especially with importance-aware data

imatrix:
  calibration / importance information is recommended for lower-bit quants
```

The practical lesson for WatchLM:

```text
Post-conversion uncalibrated Core ML palettization is weaker evidence than a
calibrated per-tensor or per-channel recipe. If a component fails under Core ML
kmeans int4, do not assume all 4-bit schemes would fail; assume this specific
post-conversion recipe is unsafe for that component.
```

llama.cpp's quantizer also exposes per-tensor overrides such as keeping output
or token embeddings at a different type and applying tensor-name overrides.
This supports our current architecture choice: policy files should keep
precision decisions explicit by tensor family and layer, rather than only
supporting one global int4/int8 knob.

## AWQ / Activation-Aware Prior

AWQ's core prior is that salient weights should be identified from activation
statistics. It protects behavior through activation-aware scaling and reports
that a small salient subset can account for much of the quantization error.

For WatchLM this means:

```text
Weight-only k-means is a useful first probe, but the production path needs a
calibration loop:
  prompt set -> collect activations/logit drift -> tensor sensitivity ranking
  -> mixed precision policy -> Core ML conversion -> Swift benchmark
```

GPTQ and SmoothQuant point in the same direction from different angles:
calibration and distribution/outlier handling matter more than arbitrary layer
selection.

## Core ML Prior

Core ML palettization supports low-bit LUT-backed weights and warns that a
single per-tensor LUT can introduce high approximation error for large
matrices. Per-grouped-channel palettization gives multiple LUTs, and
per-channel scale can help outlier-heavy rows, but local Core ML compilation
currently blocks the per-channel-scale variant for our graph.

This keeps two tracks open:

```text
Compiler-compatible path:
  per-tensor or no-scale grouped-channel Core ML palettization

Research path:
  PyTorch-side calibrated / activation-aware transform before Core ML export
```

## Mapping To MiniCPM5-1B

MiniCPM5-1B has no MoE routed experts, so the likely high-sensitivity tensors
are closer to ordinary dense transformer practice:

```text
Protect by default:
  embedding
  lm_head / output projection
  norms
  FFN until calibrated evidence improves
  edge layers

Explore first:
  middle attention projections
  Q/K/O and V as separate subgroups if a window drifts

Treat separately:
  KV cache precision, because it is runtime state rather than static weights
```

This matches current local evidence:

```text
FFN int4:
  layer0 FFN-only:  fails
  layer12 FFN-only: fails

Attention int4:
  layer12 attention-only: passes single-prompt teacher smoke
  layer10-13 attention window: fails
  layer11-12 attention window: fails
  layer11 attention-only: passes single-prompt teacher smoke
  layer11-12 grouped-channel no-scale: fails
  layer11-12 Q/K/O-only: fails
  layer11-12 Q/K-only: passes batch10 cap2 at fp16 parity
  layer11-12 O-only: passes smoke but regresses batch10 cap2
  layer11-12 V-only: passes
  layer10-13 V-only: passes
  layer8-15 V-only: matches fp16 on batch10 cap2
```

## Updated Search Rules

1. Do not widen a component class after a prefix-2 failure.

2. Do not promote a candidate based on final token agreement alone. Prefix
   top-k agreement is the early warning signal.

3. Do not keep testing FFN-wide int4 under uncalibrated Core ML palettization
   until a calibrated/groupwise route exists.

4. Treat single-layer attention success as a local clue, not a global recipe.
   Layer11 and layer12 both pass individually, while layer11-12 fails. This
   points toward accumulation or projection interaction, not a simple
   "safe layer" rule.

5. Prefer community-proven quantization ideas before adding more local
   experiments:

```text
importance/calibration data
mixed precision by tensor class
higher precision for output/embedding/norms
separate treatment of attention and FFN
projection-level attribution inside attention
separate treatment of KV cache
```

## Consequence For Next Work

After layer11-12 failed and layer11 passed, the most useful next experiments
should not be another blind widening pass. They should be architecture-led:

```text
calibrated sensitivity scoring
per-projection Q/K/O/V attribution
groupwise or importance-aware int4
KV cache precision as a separate runtime-state experiment
```

The grouped-channel Core ML retry has now been tested. It compiles when
`enablePerChannelScale` is disabled, but it still fails at prefix 2 for the
layer11-12 attention window. The immediate next step is therefore projection
attribution, not another layer-window expansion.

Projection attribution has now found a positive axis: V-only int4 across
layer11-12 preserves teacher smoke and prefix top-5 membership, while Q/K/O-only
collapses at prefix 2. The next expansion should be V-only, not whole attention.

The layer10-13 V-only expansion also preserves teacher smoke and prefix top-5
membership. This strengthens V-only as the first attention subcomponent worth
widening before batch-prompt promotion.

The layer8-15 V-only expansion matches fp16 on the batch10 cap2 gate. This is
useful evidence, but it is not enough for Watch SE deployment because V-only
compresses too little of the model. The next priority is to find another safe
ingredient or introduce calibration/importance scoring for larger components.

The QK-vs-O split adds one more clue: layer11-12 QK-only also matches fp16 on
the current batch10 cap2 gate, while O-only regresses `watch-utility-001`. This
argues for keeping O protected and treating QK as a small candidate ingredient,
not as a broad attention recipe.

Parallel to this, WatchLM should plan a calibrated quantization path rather than
relying only on Core ML post-conversion palettization:

```text
representative prompt calibration set
per-tensor or per-channel error metrics
teacher logits / top-k agreement scoring
policy generation from measured sensitivity
```
