# Qwen SE2 Simulator Asset-Store Smoke

Date: 2026-06-01
Branch: `codex/qwen-watch-se-runtime`
Target: Apple Watch SE (44mm) (2nd generation) simulator
Runtime: watchOS 26.2 simulator

## Scope

This run validates the watchOS-side asset-store and manifest path for the
current Qwen3-0.6B stateful-step context256 candidate. It does not load the
571 MB real Core ML model yet, and it does not represent physical Watch SE2
memory, Neural Engine, or thermal behavior.

## Candidate Under Test

```text
model: Qwen/Qwen3-0.6B
graph: stateful-step-kv
context: 256
artifact policy: fp32 compute + int8 storage
artifact layout: Application Support / Models/Qwen3/
stateful runtime requirement: watchOS 11+
```

The test creates a small stand-in `.mlpackage` with the same manifest-relative
path as the real Qwen artifact, writes a tokenizer fixture, saves the manifest,
verifies hashes, selects the SE2 context256 artifact, and checks that the
watchOS 11 stateful route reports `.installed`.

## Command

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild test \
  -workspace .swiftpm/xcode/package.xcworkspace \
  -scheme WatchLM-Package \
  -destination 'platform=watchOS Simulator,name=Apple Watch SE (44mm) (2nd generation)' \
  -only-testing:WatchLMCoreTests/WatchSimulatorAssetStoreXCTests/testQwenStatefulAssetStoreLayout
```

## Result

```text
Test Case '-[WatchLMCoreTests.WatchSimulatorAssetStoreXCTests testQwenStatefulAssetStoreLayout]' started.
WATCHLM_XCTEST_QWEN_ASSET_STORE state=installed context=256 graph=stateful-step-kv
Test Case '-[WatchLMCoreTests.WatchSimulatorAssetStoreXCTests testQwenStatefulAssetStoreLayout]' passed (0.015 seconds).
Executed 1 test, with 0 failures
** TEST SUCCEEDED **
```

## Current Interpretation

The SE2 simulator can build and execute the Swift package test bundle through
Xcode, and the Qwen stateful asset-store contract is now testable with
`-only-testing`.

The next SE2 step is a real model-load experiment: stage the compiled Qwen
stateful `.mlmodelc` or `.mlpackage` into the simulator-accessible container,
then run a load-only Core ML gate before attempting token generation on
simulator or physical hardware.
