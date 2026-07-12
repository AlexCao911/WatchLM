# Stateful KV Swift Runtime Contract

Date: 2026-05-30

## Scope

This note records the Swift-side runtime contract for a future MiniCPM5 Core ML
stateful KV artifact. It is separate from memory benchmark notes and from the
artifact conversion policy.

## Graph Interface

Swift now accepts two Core ML graph interfaces:

```text
logits-layered-kv
stateful-kv
```

`logits-layered-kv` is the current real artifact shape. It exposes explicit
`past_key_N`, `past_value_N`, `new_key_N`, and `new_value_N` tensors.

`stateful-kv` is the target memory-reduction shape. It exposes only token/mask
inputs and logits outputs:

```text
prefill inputs:  input_ids, position_ids, causal_mask
prefill outputs: logits
decode inputs:   token_id, position_id, causal_mask
decode outputs:  logits
```

The KV tensors live in Core ML `MLState` instead of appearing as public model IO.

## Runtime Rules

The Swift runtime can construct a `stateful-kv` bundle and validates only the
token/logits IO listed above.

At execution time, the stateful route:

1. Creates one `MLState` with `MLModel.makeState()`.
2. Runs prefill with `prediction(from:using:)`.
3. Samples from prefill logits.
4. Runs each decode step with the same `MLState`.
5. Samples from decode logits and streams tokens as before.

Because Core ML state is tied to the model instance, `stateful-kv` currently
requires prefill and decode URLs to reference the same shared Core ML model.
This is intentional: sharing one model is also the path that avoids duplicate
prefill/decode weights on Watch SE-class devices.

## Availability Gate

The local Xcode SDK gates `MLState` prediction at:

```text
macOS 15.0
iOS 18.0
watchOS 11.0
tvOS 18.0
visionOS 2.0
```

If a manifest declares `stateful-kv` on a runtime without `MLState`, the assembler
now rejects the route instead of falling back to explicit KV. That fallback would
be invalid because a stateful graph does not expose explicit KV tensors.

## Remaining Artifact Work

This commit does not create the real stateful MiniCPM5 Core ML artifact. It makes
the Swift side ready to consume one:

- manifest validation accepts `stateful-kv`
- bundle IO validation accepts token/logits-only graph IO
- runtime has an `MLState` prefill/decode loop
- assembler records and rejects unsupported stateful routes

The next conversion step is to emit a shared stateful Core ML program whose
prefill/decode path uses Core ML state buffers for all 24 MiniCPM5 layers.
