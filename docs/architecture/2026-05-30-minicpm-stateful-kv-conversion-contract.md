# MiniCPM Stateful KV Conversion Contract

Date: 2026-05-30

## Scope

This note records the conversion-side contract for the MiniCPM5 Core ML
`stateful-kv` route.

It is separate from the Swift runtime route note and benchmark notes. The runtime
note decides when stateful KV is allowed. Benchmark notes measure memory and
quality. This note defines what the real converted graph must expose.

## Why This Exists

The context-256 global-int4 explicit-KV candidate proved that the artifacts can
compile for watchOS, but host load memory still reached multi-GB RSS. The biggest
runtime risk is the explicit decode graph shape:

- prefill and decode are separate full-model graphs
- decode exposes 48 KV tensors as normal inputs and 48 KV tensors as outputs
- Core ML may expand palettized weights and allocate large execution buffers at
  load time

The stateful route is the first concrete step toward reducing that runtime shape.
It targets a single shared Core ML program with KV held in Core ML state instead
of Swift-managed tensor IO.

## Graph Interface

The `stateful-kv` conversion path exposes these public inputs:

```text
input_ids
position_ids
causal_mask
```

It exposes only one public output:

```text
logits
```

KV tensors are not public graph IO. They are Core ML states.

## MiniCPM5 State Layout

For MiniCPM5-1B at context 256, the conversion schema is:

```text
layer count:    24
KV heads:       2
head dimension: 128
state tensors:  48
state dtype:    float16
state shape:    [1, 2, 256, 128]
```

Each layer owns two state tensors:

```text
past_key_0
past_value_0
...
past_key_23
past_value_23
```

The schema is emitted into the conversion report as `graphSchema` so Swift
manifests do not have to guess which IO contract the Python conversion produced.

## Conversion Target

The conversion path uses `ct.StateType` and therefore requires the stateful Core
ML deployment target exposed by coremltools:

```text
minimum_deployment_target: ct.target.iOS18
```

Local SDK inspection shows the corresponding runtime APIs are available on
watchOS 11. The Swift runtime keeps this route gated so watchOS 10 artifacts do
not silently pretend to support stateful KV.

## Current Status

Implemented now:

- CLI graph choice includes `stateful-kv`
- conversion schema emits public inputs, public outputs, and all Core ML state
  tensors
- PyTorch wrapper registers one key and value state buffer per MiniCPM layer
- state schema handles explicit `num_key_value_heads` and Llama-like fallback to
  `num_attention_heads`
- Node contract tests verify the 24-layer, 48-state MiniCPM schema

Not proven yet:

- a full MiniCPM5-1B `stateful-kv` `.mlpackage` conversion
- dynamic prefill/decode shape behavior after tracing
- Core ML state mutation semantics for all 24 layers
- logits agreement against the PyTorch teacher
- watchOS 11 compile and physical Watch SE2/SE3 load memory

## Next Gates

1. Convert a small-context real `stateful-kv` MiniCPM artifact.
2. Validate logits and state update behavior against PyTorch.
3. Compile the artifact for watchOS 11.
4. Run host load-only and prompt smoke benchmarks against the stateful graph.
5. If memory drops, repeat with context 256 and then move to physical Watch
   SE2/SE3 testing.
