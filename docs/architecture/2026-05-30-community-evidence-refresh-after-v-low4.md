# Community Evidence Refresh After V-Low4

Date: 2026-05-30

## Scope

This note is a follow-up to the activation-guided V-low4 Core ML benchmark. It
keeps the next search step tied to outside evidence and architecture priors
instead of widening layers blindly.

## Sources Checked

- AWQ paper: https://arxiv.org/abs/2306.00978
- llama.cpp quantize README: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- llama.cpp imatrix discussion: https://github.com/ggml-org/llama.cpp/discussions/5006
- Core ML palettization overview: https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
- Core ML Tools 8 optimization notes: https://apple.github.io/coremltools/docs-guides/source/opt-whats-new.html
- antirez/ds4 README: https://github.com/antirez/ds4
- official MiniCPM5-1B GGUF repo: https://huggingface.co/openbmb/MiniCPM5-1B-GGUF
- MLX MiniCPM5-1B 4-bit repo: https://huggingface.co/mlx-community/MiniCPM5-1B-4bit
- MLX MiniCPM5-1B OptiQ 4-bit repo: https://huggingface.co/mlx-community/MiniCPM5-1B-OptiQ-4bit
- int4 KV cache on Apple Silicon: https://arxiv.org/abs/2605.05699

## Evidence

AWQ gives the strongest theoretical prior for our current direction. Its key
claim is that salient channels should be identified from activation statistics,
not raw weight magnitude, and that protecting a small fraction of salient weights
can greatly reduce low-bit error. That supports the collector we just added, but
it also says our current use is still shallow: we are selecting whole tensors,
not channel groups or scale-protected channels.

llama.cpp's quantization tooling has converged on the same idea from an
engineering direction. Its `--imatrix` path uses calibration statistics to
improve low-bit quantization, and the README examples combine imatrix with
per-tensor selectors such as `attn_v`, `ffn_down`, output tensor type, embedding
type, and regex layer overrides. That maps directly to our mixed policy schema:
component selection and layer windows are expected control surfaces.

The llama.cpp imatrix discussion is useful because it lowers the fear of tiny
calibration-set overfitting. The argument is that imatrix stores diagonal
activation expectation data and uses it as a weighted RMSE term across many
weights per entry. For us, this means the calibration collector should move
toward per-input-channel statistics and weighted quantization error, not only
module-level energy totals.

Apple's Core ML docs point to a more concrete conversion route. Plain per-tensor
palettization can have high approximation error on large matrices. Newer Core ML
optimization APIs support `per_grouped_channel`, per-channel scale, vector
palettization, sensitive k-means, and GPTQ-style calibration-data compression.
Our current policy uses per-tensor k-means palettization with no per-channel
scale, so it is deliberately conservative but not close to the best available
Core ML quantization surface.

The ds4 project is not directly portable to WatchLM because it is a specialized
engine for a specific DeepSeek V4 Flash tensor layout and quantization mix. Its
architecture is still instructive: it uses asymmetric protection. The majority
space is routed MoE experts, so ds4 quantizes those aggressively while leaving
shared experts, projections, routing, and other sensitive components untouched.
For dense MiniCPM5-1B the equivalent idea is not "quantize everything to int4";
it is "find the majority-space tensor families that tolerate compression while
guarding small but sensitive control surfaces."

MiniCPM5-specific community artifacts give useful scale targets. The official
GGUF repo ships F16 at about 2.17 GB, Q8_0 at about 1.15 GB, and Q4_K_M at about
688 MB. The MLX community 4-bit artifact reports about 608 MB. The MLX OptiQ
mixed-precision artifact is especially relevant: it reports 875 MB, 5.81
bits-per-weight, calibration across 40 mixed samples, and keeps 67 of 169
linears at 8-bit while using 4-bit for 102 linears. That is strong evidence that
a quality-preserving route may need mixed 4/8-bit, not pure int4.

The recent Apple Silicon KV-cache paper is a separate but relevant direction. It
reports int4 KV cache with fused kernels, persistent memory compression, and
quality preservation on 1B-class models. Core ML stateful graphs may not expose
the same custom-kernel freedom on watchOS, but it changes our priority: KV cache
compression should be treated as a first-class memory experiment once the weight
artifact is below the deploy gate.

## Priors For Next Experiments

1. Use activation data at channel or group granularity.
   Module-level energy is good for ranking candidates, but AWQ/imatrix both
   suggest that the actual compression objective should weight quantization
   error by activation importance per input channel or group.

2. Prefer mixed 4/8-bit over global int4.
   The OptiQ MiniCPM5 artifact and our own global-int4 collapse point in the same
   direction. The next deployable candidate should likely be "mostly int4,
   sensitive linears int8/fp16", not "everything int4."

3. Treat `attentionV` as the first safe attention axis, not the whole solution.
   Our measured V-only candidates pass quality gates, but they save too little
   memory alone. V can be combined with safer MLP subfamilies only after
   calibration ranks those subfamilies.

4. Keep Q/K/O protected until there is stronger evidence.
   Local experiments showed Q/K/O drift is much worse than V. Community evidence
   also treats per-tensor overrides as normal, so protecting Q/K/O is an
   acceptable architecture choice.

5. Move from per-tensor k-means to grouped-channel or sensitive k-means.
   Apple's docs explicitly call out per-tensor approximation error on large
   matrices. The next Core ML conversion branch should test grouped-channel int4
   where watchOS 11 compatibility permits it, and compare against the current
   per-tensor artifact with the same Swift gate.

6. Use the official Q4_K_M and MLX 4-bit sizes as deployment targets, not runtime
   architecture targets.
   They prove MiniCPM5-1B can exist around 600-700 MB in low-bit formats, but
   GGUF/MLX do not directly solve the WatchLM Core ML runtime contract. They are
   target numbers and policy inspiration.

## Concrete Next Experiment Order

1. Add channel/group activation statistics to the calibration report.
   Output per linear tensor: input-channel squared activation mean, max, entropy,
   and suggested protected-channel budget.

2. Add a policy generator for mixed int4/int8/fp16 using budgets.
   First target: match the OptiQ shape conceptually, with many MLP tensors int4,
   selected sensitive tensors int8/fp16, embeddings/lm_head/norms protected.

3. Add a Core ML grouped-channel palettization candidate.
   Gate it separately because watchOS 11 is likely the minimum deployment target
   for the newer compression mode.

4. Add weighted reconstruction checks before full Core ML export.
   For each candidate tensor family, compute activation-weighted quantization
   error from the cached calibration stats. Reject policies that look bad before
   spending time on multi-GB Core ML conversion.

5. Keep the Swift gate as promotion authority.
   A candidate is not promoted unless it passes prefix sensitivity, batch10
   teacher agreement, watchOS compile, and memory/load probes.

## Current Decision

The activation-guided V-low4 experiment has value because it validates the
calibration-guided direction. It should not be expanded by randomly picking more
layers. The next meaningful architecture move is:

```text
calibration statistics -> weighted tensor/group sensitivity -> mixed 4/8/fp16
policy -> Core ML artifact -> Swift prefix/batch/watchOS gates
```

That gives us a principled path from the current 2.0 GB sensitivity probes
toward the 600-900 MB community target range while preserving quality.
