# Stateful Shared Artifact Manifest Gate

Date: 2026-05-30

## Scope

This note records the manifest rule added after the context-256 int4 memory
diagnosis.

It is separate from the benchmark notes. The benchmark notes explain the
observed memory behavior. This note explains the runtime contract now enforced
by Swift and Node validation.

## Problem

The split context-256 int4 route compiled for watchOS, but it still carried two
full Core ML programs:

```text
prefill compiled: 516 MB
decode compiled:  516 MB
```

Host measurements showed that this split route can peak around 3 GB to 3.6 GB
RSS because Core ML materializes each program and its execution buffers at load
time.

For Watch SE2, a stateful Core ML route must not accidentally point at separate
prefill and decode packages. That would silently reintroduce the high-memory
double-model path.

## Rule

When `runtime.graphSchema.interface` is either:

```text
stateful-kv
stateful-step-kv
```

the manifest must use the same artifact path for prefill and decode:

```text
asset.prefillPath == asset.decodePath
asset.variants[N].prefillPath == asset.variants[N].decodePath
```

This matches the Swift runtime contract: stateful Core ML inference uses one
shared `MLModel` instance and one `MLState`.

## Why This Matters

The current SE2 deploy direction is:

```text
Tokenizer
-> shared stateful Core ML graph
-> MLState KV cache
-> logits processor
-> sampler
-> streaming decode
```

The older split route remains useful for diagnostics, but it is not a credible
SE2 default because it duplicates model weights and exposes explicit KV decode
IO.

## Enforcement

Swift:

```text
Sources/ModelRuntime/Model/ModelManifest.swift
```

Node validation:

```text
tools/validation/modelManifest.js
```

Tests:

```text
Tests/WatchLMCoreTests/ModelManifestTests.swift
test/modelManifest.test.js
```

## Next Implication

Any future Watch SE2 context-256 candidate manifest should declare
`stateful-step-kv` and point the SE2 `256` variant at one shared compiled
artifact. Quantization experiments can still vary precision policy, but they
should not promote a split prefill/decode manifest as the SE2 path.
