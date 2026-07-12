# Swift Benchmark CLI Prompt Selection

## Scope

This note records the `WatchLMBenchmark --prompt-ids` work as a standalone benchmark-tooling step. The goal is to make small, representative benchmark batches possible without relying on fixture prefix order.

## Why This Is Separate

The main int4/KV foundation plan is already carrying the model conversion and runtime chain history. Prompt selection is a benchmark orchestration concern, so it belongs under `docs/benchmarks/` with the other benchmark evidence notes.

## Change

- Added `--prompt-ids` to `WatchLMBenchmark`.
- Preserved the requested prompt order instead of forcing fixture order.
- Kept teacher sidecar filtering and max-token capping compatible with explicit prompt-id batches.
- Added a Swift command test for explicit prompt-id selection.

## Verification

```text
red test: RuntimeBenchmarkCommand rejected --prompt-ids as an unknown argument
swift test --filter runtimeBenchmarkCommandCanSelectPromptIDsInRequestedOrder: 1 test passed
swift run WatchLMBenchmark --runtime mock --prompt-ids watch-utility-001,zh-short-001 ...: prompts 2/2
swift test: 83 tests passed
```

## Result

Benchmark batches can now be designed by prompt id rather than only by prefix. This lets the next matrix select one or two prompts from each category while keeping runtime manageable on host and watch targets.

## Next Use

Use `--prompt-ids` to build category-balanced batches such as:

```text
zh-short-001,en-short-001,code-fix-001,watch-utility-001,safety-refusal-001
```
