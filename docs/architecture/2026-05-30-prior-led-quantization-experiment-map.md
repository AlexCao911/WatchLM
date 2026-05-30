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

## Updated Interpretation

The layer11-12 attention failure is no longer best explained as:

```text
"we used per-tensor LUT, so just make the LUT grouped"
```

The grouped-channel no-scale result still collapses at prefix 2. That means the
next high-value question is projection attribution:

```text
Which subprojection actually causes the adjacent-layer failure?
```

This is a narrower and more useful question than testing another layer window.

## Next Experiments

1. Q/K/O-only layer11-12 int4

Question:

```text
Can we compress query/key/output projections across adjacent middle layers while
leaving V in fp16?
```

Why this is useful:

```text
It separates attention-score and output-mixing error from value-state error.
```

2. V-only layer11-12 int4

Question:

```text
Does value projection quantization alone cause the prefix-2 collapse?
```

Why this is useful:

```text
KV/value paths feed the recurrent state. If V-only fails while Q/K/O passes,
KV-related precision should stay protected longer.
```

3. Activation-aware sensitivity scorer

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

4. Calibrated PyTorch-side palettization

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

5. KV cache precision as a separate runtime-state experiment

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

Run projection attribution before any more layer-window expansion:

```text
layer11-12 Q/K/O-only int4
layer11-12 V-only int4
```

If both fail, pause attention windowing and build the activation-aware scorer.
If one passes, use that result to define the next compression boundary.
