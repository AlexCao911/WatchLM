# Qwen SE2 Simulator Stateful Decode Gate

Date: 2026-06-01
Branch: `codex/qwen-watch-se-runtime`
Target: Apple Watch SE (44mm) (2nd generation) simulator
Runtime: watchOS 26.2 simulator

## Scope

This run validates the real Swift/CoreML Qwen stateful-step decode path on the
SE2 simulator. It exercises:

```text
Qwen3ChatTemplate
-> MiniCPMBytePairTokenizer with Qwen special-token settings
-> CoreMLPrefillDecodeRuntime
-> Core ML stateful-step prompt scan
-> logits sampler
-> decode loop
-> tokenizer decode
```

It is still a simulator result. It does not prove physical Watch SE2 memory,
thermal behavior, Neural Engine scheduling, or jetsam risk.

## Candidate Under Test

```text
model: Qwen/Qwen3-0.6B
graph: stateful-step-kv
context: 256
artifact policy: fp32 compute + int8 storage
compiled artifact: artifacts/coreml/compiled-watchos11-qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlmodelc
compiled size: 571 MB
tokenizer: artifacts/hf/Qwen3-0.6B/tokenizer.json
tokenizer size: 11 MB
```

## Gate

The XCTest is skipped by default. For local simulator experiments it can be
triggered by creating this sentinel file:

```text
/private/tmp/watchlm-run-qwen-se2-decode
```

The gate uses the same prompt family as the existing Qwen golden-token fixture:

```text
Turn this into a concise watch notification: The model asset finished installing and is ready for offline use.
```

Expected generated token IDs:

```text
[785, 1614, 9329, 374]
```

Expected text:

```text
The model asset is
```

## Command

```sh
touch /private/tmp/watchlm-run-qwen-se2-decode

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild test \
  -workspace .swiftpm/xcode/package.xcworkspace \
  -scheme WatchLM-Package \
  -destination 'platform=watchOS Simulator,name=Apple Watch SE (44mm) (2nd generation)' \
  -only-testing:WatchLMCoreTests/WatchSimulatorAssetStoreXCTests/testQwenRealCoreMLStatefulStepDecodeSmoke
```

## Result

```text
WATCHLM_XCTEST_QWEN_REAL_DECODE result=generated tokens=785,1614,9329,374 text="The model asset is" load_ms=91.614 first_token_ms=2438.246 decode_tps=31.21
Test Case '-[WatchLMCoreTests.WatchSimulatorAssetStoreXCTests testQwenRealCoreMLStatefulStepDecodeSmoke]' passed (3.457 seconds).
Executed 1 test, with 0 failures
** TEST SUCCEEDED **
```

Core ML also logged:

```text
[coreml] Failed to get the home directory when checking model path.
```

That warning did not prevent stateful decode in the simulator run.

## Interpretation

The Qwen branch now has simulator evidence for the full local inference chain:
template, tokenizer, stateful Core ML graph, KV state through `MLState`, logits
sampling, and text decode. The SE2 simulator produced the same 4-token sequence
as the current Qwen fp32-compute int8 golden fixture.

The `load_ms=91.614` number should be treated as a warm-cache simulator load.
The colder load-only gate recorded about 8.1 seconds for the same 571 MB
compiled model. The decode speed here, about 31 tokens/second, is useful only as
a simulator sanity signal; physical SE2/SE3 speed, memory, and thermal behavior
still need a device run.

The next gate should package or stage the same artifact for a physical SE2 test
and collect:

```text
model install/stage result
load latency
first-token latency
decode tokens/sec
resident memory / jetsam behavior
thermal state
generated token agreement
```
