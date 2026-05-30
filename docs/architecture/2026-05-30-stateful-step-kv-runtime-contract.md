# Stateful Step KV Runtime Contract

Date: 2026-05-30

## Scope

This note defines the `stateful-step-kv` Core ML runtime contract.

It is separate from benchmark evidence. The benchmark note records artifact size,
latency, memory, and quality. This document records the graph and Swift runtime
shape that made the next real Core ML artifact possible.

## Motivation

The explicit context-256 Core ML route used separate prefill and decode models.
That duplicated MiniCPM5 weights and exposed 48 full KV tensors at the Core ML
boundary. The first `stateful-kv` graph removed explicit KV IO, but the real
MiniCPM graph generated dynamic state slice updates that Core ML rejected at
execution-plan build time.

`stateful-step-kv` changes the graph to a single-token program:

```text
input_ids:    [1, 1]
position_ids: [1, 1]
causal_mask:  [1, 1, 1, context + 1]
states:       past_key_N / past_value_N, [1, 2, context, 128]
output:       logits
state update: full-state sliding write
```

## Swift Runtime Behavior

The Swift runtime treats `stateful-step-kv` as one shared stateful Core ML model.
`prefillModelURL` and `decodeModelURL` must resolve to the same model.

Prompt prefill is built by calling the same graph once per prompt token. Each
call writes that token into Core ML state. Decode then feeds one generated token
at a time through the same graph, samples from logits, and lets Core ML state
hold the sliding KV window.

This is a real inference chain:

```text
Tokenizer
  -> stateful-step prompt token loop
  -> Core ML MLState KV store
  -> logits processor
  -> sampler
  -> decode loop
  -> streaming tokens and benchmark metrics
```

## Conversion Behavior

The conversion wrapper uses registered Core ML state buffers:

```text
past_key_0 ... past_key_23
past_value_0 ... past_value_23
```

The first attempted full-state update used `copy_()`, but coremltools rejected
that trace with:

```text
ValueError: No matching select or slice.
```

The accepted form is explicit full-slice assignment:

```text
state[:, :, :, :] = concat(state[:, :, 1:, :], new_token_kv)
```

This gives coremltools a tensor assignment pattern it can lower into stateful ML
Program state writes.

## Manifest And Benchmark Contract

The manifest graph interface now accepts:

```text
logits-layered-kv
stateful-kv
stateful-step-kv
```

For `stateful-step-kv`, decode token and position input names are:

```text
input_ids
position_ids
```

The benchmark CLI accepts:

```text
--coreml-graph-interface stateful-step-kv
```

Because the same model path is used for prefill and decode, benchmark artifact
size accounting now de-duplicates identical model paths when calculating total
artifact bytes.

## Consequence

This contract trades first-token latency for memory. It removes the duplicated
prefill/decode graph shape and the explicit 48-tensor KV boundary, but prompt
prefill is now sequential. That makes it a deployability foundation, not the
final speed shape.

The next speed work should focus on:

- a faster prompt-prefill variant that still avoids duplicated weights
- active-context or shorter-prompt masks for Watch SE2
- mixed precision policy that preserves quality better than global int4
- physical Watch SE2/SE3 measurement after the host route is stable
