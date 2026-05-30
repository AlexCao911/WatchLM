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
```

llama.cpp / GGUF:

```text
https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
https://www.mintlify.com/ggml-org/llama.cpp/concepts/quantization
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
  layer11-12 V-only: passes
  layer10-13 V-only: passes
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

Parallel to this, WatchLM should plan a calibrated quantization path rather than
relying only on Core ML post-conversion palettization:

```text
representative prompt calibration set
per-tensor or per-channel error metrics
teacher logits / top-k agreement scoring
policy generation from measured sensitivity
```
