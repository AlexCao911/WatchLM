# ADR 0002: ModelRuntime Source Layout

Date: 2026-05-29
Status: Accepted

## Context

The project is moving from contracts and smoke tests toward real quantized MiniCPM5 prefill/decode inference on Apple Watch SE-class hardware. The previous layout placed runtime protocols, device profiles, model manifests, tokenizer contracts, KV cache sizing, Core ML adapters, and mock runtimes in one flat `Sources/WatchLMCore` folder. That made ownership blurry and encouraged unrelated runtime concerns to change together.

Host-side JavaScript tooling was also split across `src/`, `bin/`, `fixtures/`, `conversion/`, and `scripts/`, even though those files represent three clear workflows: conversion, benchmarking, and validation.

## Decision

Accept the proposed architecture direction with constraints.

The Swift package keeps the public product and target name `WatchLMCore` for API stability, but the implementation root is now `Sources/ModelRuntime`.

Swift runtime code is organized by responsibility:

- `Core`: runtime protocols, session states, and timing values.
- `Model`: model manifests, artifact selection, and asset state.
- `Tokenizer`: chat message/template and tokenizer abstractions.
- `Runtime/CoreML`: production Core ML adapters and smoke runtimes.
- `Runtime/Mock`: deterministic runtime used by tests and app-shell work.
- `Memory`: KV-cache sizing and memory contracts.
- `Device`: Apple Watch SE device profiles and context selection.

Host tooling moves under `tools/`:

- `tools/conversion`: Core ML artifact contracts and smoke model generation.
- `tools/benchmark`: benchmark prompt/report schemas and fixtures.
- `tools/validation`: manifest validation and the local CLI.

The proposed `Decode`, `Quant`, `Eval`, `Security`, and `Common` folders are not created yet. They should appear only when real source files need those ownership boundaries.

## Rationale

This keeps the architecture aligned with the real problem:

- MiniCPM runtime work has separate concerns for tokenizer fidelity, prefill/decode execution, KV-cache memory, device gating, and model artifact validation.
- SE 2 and SE 3 support depends on manifest and device policy, not on ad hoc runtime branching.
- Core ML adapter code should be isolated from mocks so smoke tests can mature into real model integration without contaminating protocol contracts.
- Tooling for conversion, benchmark evidence, and validation is part of the product workflow and deserves explicit ownership.

`Common` is intentionally avoided for now. Shared code can move there later only if it has stable cross-domain ownership.

## Consequences

- Imports and fixture paths are more explicit.
- The repository shape now mirrors the planned MiniCPM inference pipeline.
- The public Swift API remains `WatchLMCore`, so tests and downstream callers do not need a product rename.
- Future work has clearer homes:
  - sampler and stop criteria: `Sources/ModelRuntime/Decode`
  - mixed int4/int8 policy objects: `Sources/ModelRuntime/Quant`
  - logits and benchmark comparison code: `Sources/ModelRuntime/Eval` or `tools/benchmark`
  - hash/signature verification: `Sources/ModelRuntime/Security`

## Verification

This refactor must preserve:

- `node --test`
- `swift test`
- `xcodebuild test -scheme WatchLM -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'`
