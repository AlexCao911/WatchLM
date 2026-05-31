# Qwen SE2 Simulator Core ML Load Gate

Date: 2026-06-01
Branch: `codex/qwen-watch-se-runtime`
Target: Apple Watch SE (44mm) (2nd generation) simulator
Runtime: watchOS 26.2 simulator

## Scope

This run validates that the current Qwen3-0.6B context256 stateful-step Core ML
artifact can be loaded by a watchOS XCTest process on the SE2 simulator. It is a
load-only gate: it does not run tokenizer, decode, logits sampling, or token
generation, and it does not represent physical Watch SE2 memory pressure,
thermal behavior, or jetsam risk.

## Candidate Under Test

```text
model: Qwen/Qwen3-0.6B
graph: stateful-step-kv
context: 256
artifact policy: fp32 compute + int8 storage
compiled artifact: artifacts/coreml/compiled-watchos11-qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlmodelc
compiled size: 571 MB
```

## Test Gate

The XCTest is skipped by default. For local simulator experiments it can be
triggered by creating this sentinel file:

```text
/private/tmp/watchlm-run-qwen-se2-load
```

The test derives the repository root from `#filePath` and then loads the
compiled Qwen `.mlmodelc` via `MLModel(contentsOf:configuration:)` with
`computeUnits = .all`.

## Command

```sh
touch /private/tmp/watchlm-run-qwen-se2-load

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild test \
  -workspace .swiftpm/xcode/package.xcworkspace \
  -scheme WatchLM-Package \
  -destination 'platform=watchOS Simulator,name=Apple Watch SE (44mm) (2nd generation)' \
  -only-testing:WatchLMCoreTests/WatchSimulatorAssetStoreXCTests/testQwenRealCoreMLLoadOnly
```

## Result

```text
WATCHLM_XCTEST_QWEN_REAL_LOAD result=loaded load_ms=8114.367 model=stateful-step-kv-256-int8.mlmodelc
Test Case '-[WatchLMCoreTests.WatchSimulatorAssetStoreXCTests testQwenRealCoreMLLoadOnly]' passed (8.122 seconds).
Executed 1 test, with 0 failures
** TEST SUCCEEDED **
```

Core ML also logged:

```text
[coreml] Failed to get the home directory when checking model path.
```

That warning did not prevent model loading in the simulator run.

## Current Interpretation

The SE2 simulator can build the watchOS test bundle, see the local compiled
Qwen Core ML artifact, and load the real 571 MB model successfully. This moves
the next bottleneck from "can the simulator load the artifact at all?" to "can
the watch runtime execute a decode step and keep memory under the SE2/SE3
device envelope?"

The next gate should run a minimal stateful decode step in the simulator using
the same graph IO contract as the host benchmark, then repeat on physical SE2
with memory, first-token, tokens/sec, and jetsam/thermal telemetry.
