# Community Evidence Refresh After Scorer

Date: 2026-05-30

## Scope

This note records the evidence checked after adding the Swift quantization
sensitivity scorer.

The purpose is to keep the next experiments guided by community and
architecture priors instead of trying arbitrary tensor combinations.

## Sources Rechecked

DS4:

```text
https://github.com/antirez/ds4
https://github.com/antirez/ds4/blob/main/gguf-tools/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/README.md
https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/dataset/README.md
```

llama.cpp:

```text
https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
```

AWQ:

```text
https://arxiv.org/abs/2306.00978
https://github.com/mit-han-lab/llm-awq
```

Core ML palettization:

```text
https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
https://apple.github.io/coremltools/docs-guides/source/opt-palettization-algos.html
```

## Evidence

DS4 prefers imatrix-generated GGUFs over legacy non-imatrix downloads. Its
current public docs describe a calibration dataset and runtime collection path,
not a blind low-bit sweep.

The DS4 imatrix dataset covers source review, long-context snippets, tool-call
prompts, rewriting, summarization, extraction, translation, reasoning
benchmarks, and both thinking/non-thinking assistant prefixes. The tracked
dataset is large: thousands of rendered prompts and a multi-million-token rough
estimate. The important lesson for WatchLM is not the exact corpus size, but
the coverage shape: calibration should mirror real prompt formats and special
token patterns.

DS4 collects activation statistics with the runtime itself. For routed MoE
weights it accumulates squared input activation by column and feeds that into
the quantizer. For Q4, imatrix does not change the tensor type; it changes how
quantization error is weighted while scales/codes are chosen. DS4 also
evaluates variants with greedy/top-logit behavior and target-token negative
log likelihood against reference continuations.

llama.cpp exposes the same practical pattern through tooling: imatrix is a
first-class quantization input, output and token-embedding tensor precision can
be overridden, and regex tensor-type overrides can change individual tensor
families. This is exactly the shape our policy files should keep: explicit
mixed precision by tensor family, not one global int4 knob.

AWQ gives the strongest architectural prior: salient channels are found from
activation distribution, not raw weight magnitude. It protects behavior by
scaling salient channels before low-bit weight-only quantization, and the paper
frames the target as on-device LLM deployment. That lines up with WatchLM's
constraint: keep runtime simple, but make the offline conversion smarter.

Core ML docs reinforce the same direction. Data-free k-means is easy, but low
bits can lose accuracy. Per-tensor LUTs can be high-error on large matrices,
per-grouped-channel granularity can recover some lower-bit loss, and Sensitive
K-Means uses calibration data to weight palettization toward sensitive values.

## Implications For WatchLM

The new Swift scorer should be treated as the first promotion gate, not as the
whole quantization strategy.

The next high-value work is a calibration/importance loop:

```text
WatchLM calibration prompt suite
  -> fp16 baseline diagnostics
  -> collect activation or perturbation importance
  -> generate mixed precision policy
  -> export Core ML candidate
  -> run Swift sensitivity scorer
  -> only then run slower host/watch benchmark
```

This gives us a clear reason for each experiment:

```text
If a tensor is compressed:
  because calibration marks it low-sensitivity or because it is a structurally
  safe bulk target

If a tensor is protected:
  because community priors or local scorer evidence say it shapes token
  distribution too strongly
```

## Better Next Experiments

1. Calibration prompt suite

Build a small but representative WatchLM calibration set before more broad
quantization. It should include:

```text
short Chinese instructions
short English instructions
watch utility prompts
no-think MiniCPM chat template prompts
stop-sequence / EOS-sensitive prompts
multi-token prefixes at 1, 2, 4, 8, 12, 18, 32
```

This is the WatchLM equivalent of DS4's prompt-shape coverage, scaled down to
our device and model.

2. PyTorch-side importance collector

Before exporting Core ML, run a calibration pass over the original model and
record per-linear input activation energy:

```text
sum(x[column]^2) per module
top outlier columns per module
layer/component sensitivity summary
```

This is the dense-transformer analogue of DS4 imatrix. It is also compatible
with AWQ's "activation distribution, not weight-only guessing" principle.

3. Scorer-driven candidate ranking

Generate a small set of candidates from that importance report, then use the
Swift scorer to reject early-prefix collapse before running expensive host/watch
generation:

```text
candidate A: V-only int4 where activation importance is low
candidate B: QK int4 only where prefix top-k remains stable
candidate C: FFN grouped/SKM-style palettization only on low-sensitivity layers
```

4. Keep protected tensors protected until evidence changes

For now, keep these higher precision:

```text
embedding
lm_head / output projection
norms
attention O projection
FFN layers that have not passed calibration-aware scoring
edge layers
```

The QK/V composition failure is the warning sign: a locally safe tensor family
can become unsafe when combined with another. Policy generation must consider
interaction, not only isolated pass/fail results.

## Decision

Do not continue broad manual int4 sweeps.

The next implementation step should be:

```text
calibration prompt suite + PyTorch-side activation importance JSON
```

Then use the already-committed Swift scorer as the gate that decides which
importance-guided Core ML candidates deserve real benchmark time.
