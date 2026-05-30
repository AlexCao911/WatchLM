# Stateful Step Quantization Search Strategy

Date: 2026-05-30

## Scope

This note records the current search strategy for MiniCPM5-1B quantization on
the shared `stateful-step-kv` Core ML route.

It is separate from benchmark notes. Benchmark notes record individual
experiments. This note explains why the next experiments are chosen.

## Fixed Runtime Direction

The runtime direction is no longer the split prefill/decode route:

```text
Tokenizer
-> shared stateful-step Core ML graph
-> Core ML MLState KV cache
-> Swift logits processor
-> Swift sampler
-> streaming decode
```

The split `prefill-kv + decode` route can compile, but it duplicates a full
model and reached multi-GB host RSS. The shared stateful-step route is the only
current route worth promoting toward Watch SE2/SE3.

## Search Principle

The quantization search should be guided by logits stability, not by blind
artifact size trials.

External priors are tracked separately:

```text
docs/architecture/2026-05-30-community-quantization-priors.md
```

Each candidate must pass increasingly stronger evidence gates:

```text
1. Selector gate:
   The policy must select exactly the intended layer/component weights.

2. Prefix gate:
   Compare candidate top-k logits against fp16 at short and full prompt
   prefixes. A candidate that diverges at prefix 2 should not be widened.

3. Teacher smoke gate:
   Generate a small token cap against PyTorch teacher references.

4. Batch prompt gate:
   Run category-balanced prompts after a single-prompt result looks stable.

5. Device gate:
   Compile for watchOS and measure physical Watch SE2/SE3 load, memory,
   latency, thermal state, and jetsam/crash behavior.
```

## Evidence So Far

Global int4:

```text
memory shape: promising
quality:      fails immediately
decision:     not promotable
```

Layer0 whole-layer int4:

```text
quality: 0.0 token agreement
decision: edge whole-layer int4 is unsafe
```

Layer0 FFN-only int4:

```text
quality: 0.0 token agreement
decision: layer0 FFN is unsafe
```

Layer12 FFN-only int4:

```text
quality: 0.0 token agreement
decision: FFN risk is not only an edge-layer issue
```

Layer0 attention-only int4:

```text
quality: preserves first generated token, fails second token
decision: attention looks less destructive than FFN but not safe at layer0
```

Layer12 attention-only int4:

```text
quality: 1.0 token agreement on en-short-001
prefix:  high top-5 overlap with fp16
decision: middle-layer attention is the first useful int4 expansion direction
```

Layer10-13 attention-only int4:

```text
quality: 0.0 token agreement on en-short-001
prefix:  diverges at prefix 2
decision: do not widen directly from one stable attention layer to four layers
```

Layer11-12 attention-only int4:

```text
quality: 0.0 token agreement on en-short-001
prefix:  diverges at prefix 2
decision: adding layer11 to stable layer12 is unsafe or causes accumulation
```

Layer11 attention-only int4:

```text
quality: 1.0 token agreement on en-short-001
prefix:  top-5 membership matches fp16 at every tested prefix
decision: layer11 is not individually unsafe; layer11-12 failure points toward
          adjacent-layer accumulation or Q/K/O/V interaction
```

Layer11-12 grouped-channel attention-only int4:

```text
per-channel scale: blocked by local Core ML compiler verification
no-scale quality:  0.0 token agreement on en-short-001
prefix:            diverges at prefix 2
decision:          grouped LUT alone does not fix the adjacent-layer failure
```

Layer11-12 Q/K/O-only versus V-only int4:

```text
Q/K/O-only quality: 0.0 token agreement; diverges at prefix 2
V-only quality:     1.0 token agreement; top-5 membership matches fp16
decision:           the adjacent-layer attention failure is primarily in Q/K/O,
                    not in V
```

Layer11-12 QK-only versus O-only int4:

```text
QK-only quality: batch10 cap2 agreement 0.9, matching fp16
O-only quality:  batch10 cap2 agreement 0.85, regresses watch-utility-001
prefix:          both have high overlap with fp16, so batch gate is needed
decision:        QK is a plausible small ingredient; keep O protected
```

Layer10-13 V-only int4:

```text
quality: 1.0 token agreement on en-short-001
prefix:  top-5 membership matches fp16 at every tested prefix
decision: V-only is stable across a four-layer middle window
```

Layer8-15 V-only int4:

```text
quality: 1.0 token agreement on en-short-001
prefix:  high top-5 overlap; full prompt overlap is 4/5
batch:   10-prompt cap2 agreement matches fp16 at 0.9
decision: V-only is the first attention subcomponent that has batch-level
          parity with fp16 under the current gate
```

## Current Intuition

The current Core ML post-conversion kmeans palettization appears much riskier
for FFN projections than for middle-layer attention projections.

The working hypothesis is:

```text
Protect:
  embedding
  lm_head
  norms
  all FFN projections until calibrated quantization is available
  edge-layer attention until proven safe

Expand first:
  middle-layer V projections
  narrowly scoped QK projections after batch-level evidence

Split if drift appears:
  keep O separate and protected until calibrated evidence changes this
  activation-aware sensitivity scoring before retrying FFN
```

## Next Candidates

The layer11 result changes the next step. Both layer11 and layer12 attention
are individually stable, but the adjacent layer11-12 window is not. More local
layer sweeps would now be lower-signal unless they answer a sharper structural
question.

The next work should therefore pivot to evidence-led candidates:

```text
calibration set + sensitivity scorer
per-tensor / per-projection error metrics
V-only expansion as the immediate safe attention subcomponent
layer8-15 V-only + layer11-12 QK-only as the next local composition check
combine layer8-15 V-only with another safe compression axis only after an
equally strong gate exists for that axis
groupwise or importance-aware int4 before retrying FFN or wider attention
```

FFN-wide int4 should pause until we can test calibrated or groupwise
quantization before Core ML conversion. Continuing to widen uncalibrated FFN
int4 would be low-signal because both edge and middle FFN isolation failed.

## Promotion Rule

A candidate should not be called an SE2/SE3 candidate unless it satisfies all of
these:

```text
context:          256
graph:            shared stateful-step-kv
Core ML compile:  watchOS succeeds
quality:          category-balanced prompt agreement is close to fp16 teacher
memory:           physical Watch SE2/SE3 does not crash or jetsam
latency:          first-token and decode speed are inside watch usability gates
thermal:          no immediate severe thermal failure during short turns
```
