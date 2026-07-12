# Real Stateful Step KV Artifact Evidence

Date: 2026-05-30

## Scope

This note records the first real MiniCPM5-1B `stateful-step-kv` Core ML artifact
attempts.

It is separate from the runtime contract note. This document records conversion,
compilation, host inference, memory, latency, and quality evidence.

## Context 16 Smoke

Uncompressed conversion:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-16/stateful-step-kv-16.mlpackage
size: 2,162,169,171 bytes
```

Int4 package:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-16-int4/stateful-step-kv-16-int4.mlpackage
size: 541,379,093 bytes
```

Compiled artifacts:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-16-int4/stateful-step-kv-16-int4.mlmodelc
artifacts/coreml/compiled-watchos-stateful-step-kv-16-int4/stateful-step-kv-16-int4.mlmodelc
compiled size: 516 MB each
```

Host inference report:

```text
artifacts/benchmarks/stateful-step-kv-16-int4-smoke.json
prompts: 1/1
peak RSS: 634.91 MB
first token: 361.80 ms
decode throughput: 53.48 tok/s
```

## Context 256 SE2 Candidate

Uncompressed conversion:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256/stateful-step-kv-256.mlpackage
size: 2,162,169,955 bytes
```

Int4 package:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlpackage
size: 541,379,877 bytes
```

Compiled artifacts:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlmodelc
artifacts/coreml/compiled-watchos-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlmodelc
compiled size: 516 MB each
```

Host inference report without teacher reference:

```text
artifacts/benchmarks/stateful-step-kv-256-int4-smoke.json
prompts: 1/1
peak RSS: 953.09 MB
first token: 11,429.84 ms
decode throughput: 31.93 tok/s
```

Host inference report with PyTorch teacher reference:

```text
artifacts/benchmarks/stateful-step-kv-256-int4-teacher-smoke.json
prompts: 1/1
peak RSS: 651.25 MB
first token: 12,457.16 ms
decode throughput: 32.99 tok/s
token agreement: 0.0
generated token IDs: [5, 67778]
```

## Interpretation

`stateful-step-kv` changes the memory picture materially. The earlier
context-256 split int4 pair loaded at roughly 3 GB RSS on host. The real
context-256 stateful-step int4 graph loads and runs in the 650-950 MB range on
host, depending on warm state and telemetry timing.

The artifact is now a real watchOS compile candidate for context256 because the
watchOS 11 compile gate passes for the single int4 stateful graph.

The Swift package target also builds for the local Apple Watch SE 3 simulator:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild build \
  -scheme WatchLMCore \
  -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'

result: BUILD SUCCEEDED
```

It is not a production-quality SE2 candidate yet:

- global int4 still has 0.0 teacher token agreement on the smoke prompt
- first-token latency is too high because prompt prefill is one Core ML call per
  prompt token against a 256-slot state
- physical Watch SE2/SE3 memory and thermal behavior still need device runs

## Next Optimization Targets

1. Quality: replace global int4 with a mixed precision policy for the stateful
   graph, protecting embeddings, lm_head, attention, norms, and edge layers.
2. First-token latency: add a faster prompt build path or context variants that
   keep the single-model memory advantage without forcing every prompt token to
   attend over 256 state slots.
3. Device validation: install the context256 watchOS compiled artifact on Watch
   SE2/SE3 and measure load, first token, decode throughput, peak memory, and
   thermal state.
