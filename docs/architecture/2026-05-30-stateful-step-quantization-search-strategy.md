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
  middle-layer attention projections

Split if drift appears:
  Q/K/O-only vs V-only
  narrower middle windows before wider windows
```

## Next Candidates

The next candidates should widen only along the stable axis found so far:

```text
layer 10-13 attention-only int4
layer 8-15 attention-only int4
layer 12 Q/K/O-only int4
layer 12 V-only int4
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
