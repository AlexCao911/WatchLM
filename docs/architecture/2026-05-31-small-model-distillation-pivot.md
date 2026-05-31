# Small Model Distillation Pivot

Date: 2026-05-31

## Decision

MiniCPM5-1B remains the teacher, reference, and Core ML runtime stress test. It
should no longer be treated as the most likely final Watch SE2/SE3 runtime
model.

The new primary path is:

```text
teacher:        MiniCPM5-1B and stronger host models
student target: 125M-350M first, 600M only as a stretch candidate
runtime:        Swift + Core ML stateful-step KV
context:        128/256 first, 512 only after device evidence
compression:    int4/int8 plus distillation, not post-training int4 alone
```

## Why This Changed

Local evidence has become strong enough to stop over-investing in MiniCPM5-1B
post-training quantization:

```text
global stateful-step int4:
  artifact/RSS shape is closer to SE2, but token agreement collapses.

attentionV low8:
  full12 calibration gate passes, but artifact remains around 2GB.

attentionV low8 + split-FFN int4:
  converts, compiles for watchOS 11, and runs 84/84 diagnostics, but full12
  sensitivity fails with average prefill top-k overlap 0.29.
```

This means the Swift/Core ML inference chain is real enough to evaluate
candidates, but the current model scale is the bottleneck.

## External Priors

MobileLLM is the closest architectural prior for this pivot: it explicitly
targets sub-billion on-device models and emphasizes architecture choices such as
deep-thin layouts, embedding sharing, and grouped-query attention for 125M/350M
models.

TinyStories is the data prior: narrow synthetic/task-focused data can make much
smaller models useful inside constrained domains. WatchLM should use that idea
for watch-specific short-turn instructions rather than generic web-scale
language modeling.

Core ML joint compression remains useful, but as a second-stage optimizer after
choosing a viable student model. Apple documents combining pruning,
palettization, and quantization; that is a better follow-up than continuing
single-axis int4 sweeps on the 1B artifact.

References:

```text
MobileLLM: https://arxiv.org/abs/2402.14905
TinyStories: https://arxiv.org/abs/2305.07759
Core ML joint compression: https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html
```

## New Sizing Gate

Candidate profiles now go through:

```text
tools/validation/modelCandidateSizing.js
tools/validation/fixtures/model-candidates.json
```

Validation command:

```bash
node tools/validation/watchlm-validate.js candidates tools/validation/fixtures/model-candidates.json
```

SE2 planning targets:

```text
artifact <= 650MB
estimated peak RSS <= 850MB
context <= 256 for first promotion
```

SE3 planning targets:

```text
artifact <= 750MB
estimated peak RSS <= 950MB
```

These are not final device proof. They are pre-conversion gates to avoid
spending time on candidates that cannot plausibly become watch-runtime artifacts.

## Current Candidate Readout

MiniCPM5-1B V-low8:

```text
role:           teacher-baseline
artifact:       about 2157MB
peak RSS:       about 2187MB
SE2 gate:       fail
```

Distilled WatchLM 350M int4:

```text
role:           runtime-candidate
artifact:       about 229MB
peak RSS:       about 477MB
SE2 gate:       pass
```

Distilled WatchLM 600M int4:

```text
role:           stretch runtime-candidate
artifact:       about 385MB
peak RSS:       about 927MB
SE2 gate:       fail
```

## Next Engineering Work

The next useful implementation steps are:

```text
1. pick an existing 125M/350M/600M architecture candidate that can be legally
   converted and evaluated locally.
2. export a context-256 stateful-step Core ML graph for that model.
3. reuse the existing Swift tokenizer/runtime/benchmark chain against the new
   manifest identity.
4. generate watch-domain distillation data from MiniCPM5-1B teacher outputs.
5. fine-tune or distill the student, then repeat Core ML + Swift benchmark.
```

MiniCPM5-1B quantization work should continue only as a teacher/reference line
or as a narrow Core ML compression experiment. It should not block the smaller
student model path.
