# Context-256 Int4 Memory Diagnosis

Date: 2026-05-30

## Scope

This note explains the memory spike seen when loading the context-256 global-int4 Core ML candidate.

It is separate from the watchOS deploy gate note. The deploy gate answers whether the artifacts compile for watchOS. This note answers why host resident memory is still too high.

## Artifacts

Deployment-first int4 pair:

- prefill: `artifacts/coreml/real-minicpm5-prefill-kv-256-int4/prefill-kv-256-int4.mlpackage`
- decode: `artifacts/coreml/real-minicpm5-decode-256-int4/decode-256-int4.mlpackage`

Compiled macOS pair used to remove runtime `.mlpackage` compilation from the measurement:

- prefill: `artifacts/coreml/compiled-macos-prefill-kv-256-int4/prefill-kv-256-int4.mlmodelc`
- decode: `artifacts/coreml/compiled-macos-decode-256-int4/decode-256-int4.mlmodelc`

Compiled sizes:

```text
prefill compiled: 516 MB
decode compiled:  516 MB
```

## Measurements

### Full inference from `.mlpackage`

Report: `artifacts/benchmarks/context256-int4-deploy-first-smoke.json`

```text
peak resident memory: 3627.47 MB
first token: 263.66 ms
decode throughput: 1.49 tokens/sec
token agreement: 0.0
```

### Full inference from precompiled macOS `.mlmodelc`

Report: `artifacts/benchmarks/context256-int4-compiled-macos-smoke.json`

```text
peak resident memory: 3027.61 MB
first token: 214.17 ms
decode throughput: 12.15 tokens/sec
token agreement: 0.0
```

Removing runtime `.mlpackage` compilation reduced peak RSS by about 600 MB, but peak memory stayed around 3 GB.

### Load-only diagnostics

These were run with the dedicated Swift benchmark load probe:

```text
swift run WatchLMBenchmark --runtime coreml --load-only --coreml-load-target prefill ...
swift run WatchLMBenchmark --runtime coreml --load-only --coreml-load-target decode ...
swift run WatchLMBenchmark --runtime coreml --load-only --coreml-load-target both ...
```

Reports:

- `artifacts/benchmarks/context256-int4-prefill-load-only.json`
- `artifacts/benchmarks/context256-int4-decode-load-only.json`
- `artifacts/benchmarks/context256-int4-both-load-only.json`

Results:

```text
prefill-only peak RSS: 1770.47 MB
decode-only peak RSS:  2818.44 MB
both load-only peak:   2850.84 MB
```

## Interpretation

The memory spike is not caused by KV cache size alone. The context-256 KV cache is small relative to gigabytes of RSS.

The main contributors are:

1. Core ML int4/palettized weights are compact on disk, but Core ML can expand, rearrange, or cache weights and execution plans internally.
2. The current split graph has separate prefill and decode programs, so the deployment package still carries two graph copies.
3. The explicit-KV decode graph is the larger runtime memory risk. Decode-only loading reached about 2.82 GB RSS, likely because the graph includes 48 full past-KV tensor inputs and corresponding execution buffers.
4. `.mlpackage` runtime compilation adds overhead, but it is not the root cause. Precompiled `.mlmodelc` still peaked around 3.03 GB during one-prompt inference.

## Consequence For Watch SE2

The context-256 int4 pair is a real watchOS compile candidate, but it is not yet a credible Watch SE2 runtime candidate.

The next optimization should focus on decode memory, not just lower-bit weight storage:

- move from explicit 48-input KV decode toward stateful Core ML cache if watchOS/Core ML supports the required state update pattern
- reduce duplicated prefill/decode weights, ideally through a single shared/stateful program
- test a smaller decode context variant only as a fallback, because the product goal still asks for context256 on SE2
- keep quality separate from deployment: global int4 currently runs but has 0.0 token agreement on the smoke prompt

