# Qwen Installed-Root Decode Gate

Date: 2026-06-01
Branch: `codex/qwen-watch-se-runtime`
Target: Apple Watch SE 2 simulator and physical-device preparation

## Scope

This checkpoint adds a decode gate that only reads model assets from the app's
installed `Application Support/WatchLM` root:

```text
ModelAssetStore.defaultRootURL()
-> model-manifest.json
-> Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage
-> Models/Qwen3/tokenizer.json
-> CoreMLRuntimeAssembler
-> CoreMLPrefillDecodeRuntime
```

The earlier simulator decode gate intentionally read from the Mac checkout so
it could validate the real graph quickly. A physical Watch cannot use that
path. This gate is the bridge to a device-style run: if the app container has
not been staged, it skips instead of falling back to repository artifacts.

## XCTest

```text
WatchSimulatorAssetStoreXCTests.testQwenInstalledApplicationSupportStatefulStepDecodeSmoke
```

The test expects the same current Qwen golden sequence as the real Core ML
simulator decode gate:

```text
tokens: [785, 1614, 9329, 374]
text: The model asset is
termination: maxTokens
decode steps: 3
```

## Command

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild test \
  -scheme WatchLM-Package \
  -destination 'platform=watchOS Simulator,name=Apple Watch SE (44mm) (2nd generation)' \
  -only-testing:WatchLMCoreTests/WatchSimulatorAssetStoreXCTests/testQwenInstalledApplicationSupportStatefulStepDecodeSmoke
```

## Current Result

The SE2 simulator run passed with a deliberate skip because the simulator app
container had not yet been staged:

```text
Test skipped - Stage Qwen assets into /Users/alexandercou/Library/Developer/CoreSimulator/Devices/BEFDB6DB-55EC-4B2F-9878-FEE59586EFA0/data/Library/Application Support/WatchLM before running the installed Application Support decode gate.
Executed 1 test, with 1 test skipped and 0 failures
** TEST SUCCEEDED **
```

This is the expected unstaged behavior. It proves the new gate is not silently
loading the model from `artifacts/` in the repository.

## Staging Before a Real Run

Use the existing Swift staging installer to populate an installed root:

```sh
swift run WatchLMBenchmark \
  --manifest tools/validation/fixtures/qwen3-0.6b-stateful-step-model-manifest.json \
  --asset-base artifacts/runtime-candidates \
  --device-profile watch-se-2 \
  --stage-to <Application Support/WatchLM> \
  --output <staging-result.json>
```

After staging, rerun the installed-root decode gate. A successful run should
print:

```text
WATCHLM_XCTEST_QWEN_INSTALLED_DECODE ...
```

and record load latency, first-token latency, decode tokens/sec, and generated
token agreement from the installed app-container layout.

## Remaining Physical-Device Gap

```text
done: installed-root-only decode gate
done: clean skip when the app container is not staged
pending: expose a named physical Watch SE2/SE3 destination in Xcode
pending: install the watch test app
pending: stage Qwen assets into the physical watch app container
pending: run installed-root decode on the device
pending: collect memory, jetsam, thermal, first-token, and decode tok/s data
```
