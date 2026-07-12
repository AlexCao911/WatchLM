# Core ML Stateful KV Route

Date: 2026-05-30

## Scope

This note records the runtime route decision for MiniCPM5 Core ML KV cache execution.
It is separate from benchmark notes: benchmarks measure memory and quality, while this
note explains which Swift runtime path is allowed for a given OS and graph interface.

## Local SDK Evidence

The local Xcode watchOS SDK exposes Core ML stateful prediction APIs only from
watchOS 11.0:

```text
MLStateConstraint.bufferShape
@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)

MLModel.makeState()
@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)

MLModel.prediction(from:using:options:)
@available(macOS 15.0, iOS 18.0, watchOS 11.0, tvOS 18.0, visionOS 2.0, *)
```

The package still supports watchOS 10 as a deployment target. Therefore stateful KV
must remain a gated path, not the only runtime path.

## Current Artifact Reality

The current context-256 int4 artifact uses `logits-layered-kv`:

```text
prefill inputs:  input_ids, position_ids, causal_mask
prefill outputs: logits, present_key_N, present_value_N
decode inputs:   token_id, position_id, causal_mask, past_key_N, past_value_N
decode outputs:  logits, new_key_N, new_value_N
```

This is an explicit KV graph. Even on watchOS 11+, it cannot use `MLState`
automatically because the graph IO contract is not stateful.

## Route Rules

The Swift runtime now chooses a route from both manifest intent and artifact shape:

```text
kvCacheMode=stateful-preferred + graphInterface=stateful-kv + OS supports MLState
  -> statefulKV

kvCacheMode=stateful-preferred + graphInterface=stateful-kv + OS does not support MLState
  -> unsupportedStatefulKV

kvCacheMode=stateful-preferred + graphInterface=logits-layered-kv
  -> explicitSlotRing

kvCacheMode=slot-ring
  -> explicitSlotRing

kvCacheMode=contiguous-sliding
  -> explicitContiguousSliding
```

This matters because the memory spike seen in the context-256 int4 run is dominated
by the explicit Core ML decode graph, not by the raw KV cache byte count.

The unsupported stateful route is rejected during assembly. It must not silently
fall back to explicit KV, because a stateful graph has no explicit KV tensor IO.

## Consequence

To make context-256 credible on Watch SE2/SE3, the next model artifact work should
produce a real stateful or shared-weight Core ML graph. The Swift runtime can now
record whether it is still using explicit KV or has actually crossed into stateful
KV, so later benchmark reports should not blur those two cases.
