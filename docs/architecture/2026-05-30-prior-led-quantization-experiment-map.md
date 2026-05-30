# Prior-Led Quantization Experiment Map

Date: 2026-05-30

## Scope

This note records the next quantization direction after the layer11/layer12
attention isolation work.

The goal is to avoid blind layer sweeps. Each next experiment should be tied to
a community or architecture prior and should answer a specific question.

## Sources

Core ML palettization:

```text
https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
https://apple.github.io/coremltools/docs-guides/source/opt-opt1_3.html
https://apple.github.io/coremltools/source/coremltools.optimize.coreml.palettization.html
```

AWQ:

```text
https://arxiv.org/abs/2306.00978
https://github.com/mit-han-lab/llm-awq
```

llama.cpp / imatrix:

```text
https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
```

DS4:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/dataset/README.md
```

Related PTQ directions:

```text
GPTQ:        https://arxiv.org/abs/2210.17323
SmoothQuant: https://arxiv.org/abs/2211.10438
QuaRot:      https://arxiv.org/abs/2404.00456
```

## External Priors

Core ML's palettization docs say per-tensor LUTs can have high approximation
error for large matrices. The natural first check was therefore
`per_grouped_channel`, because it creates multiple LUTs across channel groups.

Core ML's OPT example also distinguishes two paths:

```text
data-free k-means palettization
calibration-data SKM palettization
```

That distinction matters for MiniCPM5-1B because uncalibrated FFN int4 has
already failed at both edge and middle layers.

AWQ's main lesson is that salient channels should be identified from activation
statistics, not from weights alone. It protects behavior through activation
aware scaling rather than by randomly choosing layers.

llama.cpp's imatrix path is a similar practical prior: low-bit quantization
works better when a representative calibration set influences tensor
importance. The useful lesson is not that we should use GGUF on watchOS; it is
that low-bit recipes need measured tensor importance.

DS4 shows an asymmetric recipe: aggressively compress the component class that
dominates size, while keeping attention projections, output head, embedding,
routers, norms, and other decision-making tensors at higher precision. MiniCPM
is dense, so there is no routed-expert bulk target. FFN is the size target, but
local evidence says uncalibrated FFN int4 is unsafe.

## Local Evidence Added

Layer11 attention-only:

```text
quality: 1.0 token agreement
prefix:  top-5 membership matches fp16 at every tested prefix
```

Layer12 attention-only:

```text
quality: 1.0 token agreement
prefix:  high top-5 overlap with fp16
```

Layer11-12 attention-only:

```text
quality: 0.0 token agreement
prefix:  diverges at prefix 2
```

Layer11-12 grouped-channel no-scale:

```text
quality: 0.0 token agreement
prefix:  diverges at prefix 2
```

Layer11-12 grouped-channel with per-channel scale:

```text
status: blocked by local Core ML compiler verification
```

Layer11-12 Q/K/O-only versus V-only:

```text
Q/K/O-only: fails at prefix 2 and teacher smoke
V-only:     passes all tested prefixes and teacher smoke
```

Layer11-12 QK-only versus O-only:

```text
QK-only: matches fp16 batch10 cap2 agreement at 0.9
O-only:  passes prefix/smoke but drops batch10 cap2 agreement to 0.85
```

Layer10-13 V-only:

```text
V-only: passes all tested prefixes and teacher smoke
```

Layer8-15 V-only:

```text
V-only: passes teacher smoke
prefix: high top-5 overlap, with one full-prompt top-5 drift
batch:  matches fp16 batch10 cap2 token agreement at 0.9
```

## Updated Interpretation

The layer11-12 attention failure is no longer best explained as:

```text
"we used per-tensor LUT, so just make the LUT grouped"
```

The grouped-channel no-scale result still collapses at prefix 2. That means the
next high-value question was projection attribution:

```text
Which subprojection actually causes the adjacent-layer failure?
```

That question now has a useful answer: Q/K/O is the sensitive side, while V is
locally safe across layers 11-12, remains stable across layers 10-13, and
matches fp16's batch10 cap2 agreement profile across layers 8-15.

The Q/K/O split has now become sharper:

```text
QK-only can pass the current batch gate.
O-only can pass the single-prompt gates while still regressing a watch utility
prompt.
Q/K/O together still fails early.
```

The working interpretation is interaction/accumulation, not a simple rule that
all Q/K/O tensors are unsafe. QK can be explored as a small ingredient; O should
remain high precision.

## Next Experiments

1. Combine layer8-15 V-only with layer11-12 QK-only

Question:

```text
Can the two locally safe attention ingredients compose without triggering the
same interaction failure as Q/K/O?
```

Why this is useful:

```text
V-only is quality-stable but too small to solve size. QK-only is the first
additional attention axis with batch parity, but composition must be tested
before widening.
```

2. Activation-aware sensitivity scorer

Question:

```text
Which tensors show the largest teacher-logit drift on representative watch
prompts when perturbed or compressed?
```

Why this is useful:

```text
This turns the search from layer guessing into an AWQ/imatrix-style importance
measurement loop.
```

3. Calibrated PyTorch-side palettization

Question:

```text
Does calibration-data palettization produce a materially better Core ML artifact
than post-conversion k-means?
```

Why this is useful:

```text
Core ML's docs provide both data-free and calibration-data routes. Our current
failures are all data-free post-conversion results.
```

4. KV cache precision as a separate runtime-state experiment

Question:

```text
Can the Swift stateful-step runtime store KV in lower precision without changing
static weight policy?
```

Why this is useful:

```text
KV cache affects memory growth with context and decode state. It should not be
mixed into static-weight compression conclusions.
```

## Current Priority

Do not widen whole attention windows. Widen only along the safe V-only axis:

```text
layer10-13 V-only int4
layer8-15 V-only int4
```

Layer8-15 V-only now passes the current batch gate at fp16 parity. The next
small local ingredient is layer11-12 QK-only. The next broad direction should
be the activation-aware sensitivity scorer for larger targets such as FFN.
