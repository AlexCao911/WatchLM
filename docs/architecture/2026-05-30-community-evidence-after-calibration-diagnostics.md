# Community Evidence After Calibration Diagnostics

Date: 2026-05-30

## Scope

This note follows the new Swift `--calibration-prompts` diagnostics path and
the low-risk V-attention smoke result.

The goal is to turn the next quantization work into evidence-led experiments,
not another blind layer/component sweep.

## Fresh Local Anchor

Swift/Core ML calibration-prefix diagnostics now run directly from:

```text
tools/benchmark/fixtures/calibration-prompts.json
```

The first smoke compared FP16 against the low-risk V-attention low8 policy:

```text
prompt limit: 2 calibration prompts
prefixes:     1, 2, 4, 8
points:       8
gate:         pass
avg top-k:    0.95
top-1:        1.0
```

Interpretation:

```text
The Swift evidence path is ready.
V-attention-only compression remains quality-safe on this smoke.
V-attention-only compression is still too small a memory win to be the deploy
strategy.
```

## Sources Rechecked

- DS4: https://github.com/antirez/ds4
- DS4 imatrix tooling: https://github.com/antirez/ds4/blob/main/gguf-tools/imatrix/README.md
- llama.cpp quantize: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- llama.cpp imatrix: https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
- AWQ: https://arxiv.org/abs/2306.00978
- GPTQ: https://arxiv.org/abs/2210.17323
- SmoothQuant: https://arxiv.org/abs/2211.10438
- Core ML palettization overview: https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html
- Core ML OPT compression example: https://apple.github.io/coremltools/docs-guides/source/opt-opt1_3.html
- Official MiniCPM5-1B GGUF: https://huggingface.co/openbmb/MiniCPM5-1B-GGUF
- MLX OptiQ MiniCPM5-1B 4-bit: https://huggingface.co/mlx-community/MiniCPM5-1B-OptiQ-4bit

## Evidence That Changes The Search

### 1. DS4 is architecture-specific, not a generic low-bit recipe

DS4 gets useful 2-bit behavior by quantizing only routed MoE experts and
leaving shared experts, projections, routing, and other sensitive tensors
higher precision. Its README also says the engine is not a general GGUF loader.

For MiniCPM5-1B:

```text
Do not copy DS4's 2-bit number.
Copy its asymmetric architecture logic.
```

MiniCPM5-1B is dense, so there is no routed-expert bulk area. The closest dense
analogue is the MLP/FFN family, but local uncalibrated FFN int4 already failed.
That means the FFN route needs better quantization mechanics, not wider blind
FFN selection.

### 2. llama.cpp uses imatrix and per-tensor controls

llama.cpp's quantize tooling recommends importance matrices for lower-bit
quality and exposes tensor-level controls such as output tensor type, token
embedding type, and regex tensor overrides.

For WatchLM:

```text
The mixed policy schema is the right control plane.
The missing piece is not more arbitrary layer lists; it is a better importance
objective per tensor/group.
```

Our current channel activation stats are a start, but the next scorer should
look more like imatrix: per tensor/group squared activation statistics,
entropy/active fraction, and activation-weighted reconstruction error.

### 3. AWQ/GPTQ/SmoothQuant all point away from raw global int4

AWQ's useful claim is that salient channels should be found from activation
statistics, not raw weight magnitude. GPTQ and SmoothQuant reach a related
conclusion from second-order error and outlier-migration angles: distribution
and calibration matter.

For WatchLM:

```text
Global int4 and uncalibrated per-tensor k-means are diagnostic probes, not the
deployment candidate.
```

The next candidate should be calibrated mixed precision:

```text
protected: embedding, lm_head, norms, output projections, last block, sensitive
           Q/K/O, high-risk gate rows
compressed: selected middle MLP tensors and already-stable V projections
```

### 4. Core ML gives a concrete alternative to our current palettization

Apple's Core ML docs call out the approximation risk of one LUT per large
tensor, introduce `per_grouped_channel`, and show that for OPT-1.3B data-free
int4 per-tensor collapses while per-block/per-grouped-channel and calibration
data improve the trade-off.

For WatchLM:

```text
Current Core ML k-means palettization is only one point in the space.
The next Core ML route should test compiler-compatible int4 per-block or
grouped-channel compression on selected tensor families.
```

Important caveat:

```text
Every new Core ML compression mode must pass watchOS compile/load gates before
we treat its quality result as actionable for SE2/SE3.
```

### 5. MiniCPM-specific artifacts give targets and a policy shape

The official MiniCPM5-1B GGUF repo reports:

```text
F16:    2.17 GB
Q8_0:   1.15 GB
Q4_K_M: 688 MB
```

The MLX OptiQ MiniCPM5-1B artifact reports a mixed 4/8-bit policy:

```text
disk:             875 MB
bits per weight:  5.81
calibration:      40 mixed samples
8-bit:            67 of 169 linears
4-bit:            102 of 169 linears
protected shape:  output projections, gate, last block, lm_head
```

For WatchLM:

```text
The realistic quality-preserving target is probably not pure int4.
It is a 600-900 MB mixed 4/8/fp16 Core ML artifact with a stricter runtime
memory strategy.
```

## Next Experiment Hypotheses

### Hypothesis A: OptiQ-shaped policy is higher value than widening V-only

Try a Core ML policy that conceptually mirrors the OptiQ shape:

```text
fp16: embedding, lm_head, norms
int8: last block, output projections, gate projections, Q/K/O in sensitive layers
int4: selected middle MLP up/down tensors plus low-risk V projections
```

Why:

```text
It follows MiniCPM-specific community evidence and targets the tensor families
that can actually move size.
```

Immediate gate:

```text
Full calibration-prefix diagnostics, not only prompt-limit 2.
```

### Hypothesis B: MLP requires better mechanics before broader selection

Do not retry naive FFN-wide int4. Instead, test one MLP subfamily with a more
granular Core ML compression mode:

```text
candidate 1: int4 per-block linear quantization, block size 32
candidate 2: int4 grouped-channel palettization, group size 32 or 16
candidate 3: calibration-data SKM palettization if the conversion stack and
             watchOS compile path support it
```

Why:

```text
Apple's OPT example shows per-tensor 4-bit can collapse while more granular
4-bit variants remain close to baseline.
```

Immediate gate:

```text
watchOS compile first, then calibration-prefix sensitivity.
```

### Hypothesis C: Channel/group weighted error should reject bad policies early

Before spending time on full Core ML export, compute:

```text
activation-weighted MSE by tensor/group
top-column concentration
active fraction
entropy
protected-channel budget
```

Why:

```text
This converts AWQ/imatrix priors into a cheap pre-export rejection gate.
```

Immediate gate:

```text
Reject candidates whose weighted error is concentrated in high-activation
groups, even if plain weight MSE looks acceptable.
```

### Hypothesis D: Calibration prompts need a 40-128 sample tier

The current 12-prompt suite is good for fast gating. Community evidence from
imatrix/AWQ/OptiQ suggests the policy search should also have a larger mixed
calibration tier.

Use two tiers:

```text
fast gate:      current 12 prompts, fixed prefixes 1,2,4,8,12,18,32
policy search: 40-128 prompts across watch utility, Chinese, English, code,
               tool-call style, stop sequences, safety, and short reasoning
```

Why:

```text
The small suite should catch early collapse; the larger suite should drive
importance statistics and reduce overfitting to two or twelve prompts.
```

## Recommended Next Order

1. Run the full 12-prompt calibration-prefix sensitivity for the current FP16
   baseline and low8 low-risk V candidate.

2. Add an activation-weighted reconstruction scorer to rank tensor families
   before Core ML export.

3. Generate one OptiQ-shaped mixed policy using the current schema:
   V low-risk int4/low8 plus MLP candidates ranked by weighted error, while
   protecting output projections, gates, last block, lm_head, embeddings, and
   norms.

4. Add one Core ML conversion variant for per-block int4 or grouped-channel
   int4 on selected MLP tensors, then gate it with watchOS compile before
   runtime quality.

5. Expand calibration prompts from 12 to a 40-128 sample search tier, keeping
   the 12-prompt suite as the quick regression gate.

## Decision

The next high-value work is not more V-only expansion. It is:

```text
community-shaped mixed policy
  + activation-weighted rejection gate
  + Core ML granular quantization mode
  + full calibration-prefix Swift diagnostics
```

This is the most direct path from the current quality-safe but too-large
artifact toward a Watch SE2/SE3 candidate that can plausibly fit while keeping
MiniCPM5-1B behavior intact.
