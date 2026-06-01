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
-> Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlmodelc
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

## Unstaged Result

The SE2 simulator run passed with a deliberate skip because the simulator app
container had not yet been staged:

```text
Test skipped - Stage Qwen assets into /Users/alexandercou/Library/Developer/CoreSimulator/Devices/BEFDB6DB-55EC-4B2F-9878-FEE59586EFA0/data/Library/Application Support/WatchLM before running the installed Application Support decode gate.
Executed 1 test, with 1 test skipped and 0 failures
** TEST SUCCEEDED **
```

This is the expected unstaged behavior. It proves the new gate is not silently
loading the model from `artifacts/` in the repository.

## Asset Format Finding

The first staged run used the uncompiled `.mlpackage` path and failed at Core
ML load:

```text
It is not a valid .mlmodelc file.
Compile the model with Xcode or `MLModel.compileModel(at:)`.
```

For the watch installed-root path, the manifest now points at the precompiled
watchOS `.mlmodelc` artifact:

```text
Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlmodelc
```

This keeps expensive model compilation out of the watch runtime path.

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

Current SE2 simulator staging result:

```text
items: 3
total_bytes: 609897300
model_sha256: 97ae982de576d323836eb05f91f7794a2efffd8e226c437a1c272aff7c49eef4
tokenizer_sha256: aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

## Staged Decode Result

After staging the compiled `.mlmodelc`, the SE2 simulator installed-root gate
ran real Core ML inference:

```text
WATCHLM_XCTEST_QWEN_INSTALLED_DECODE result=generated tokens=785,1614,9329,374 text="The model asset is" root="/Users/alexandercou/Library/Developer/CoreSimulator/Devices/BEFDB6DB-55EC-4B2F-9878-FEE59586EFA0/data/Library/Application Support/WatchLM" load_ms=3367.491 first_token_ms=964.668 decode_tps=35.28
Executed 1 test, with 0 failures
** TEST SUCCEEDED **
```

The result confirms the staged app-container layout can drive the same Swift
inference chain as the repository-path decode gate:

```text
Qwen3ChatTemplate
-> MiniCPMBytePairTokenizer with Qwen token settings
-> ModelAssetStore installed manifest
-> CoreMLRuntimeAssembler
-> stateful Core ML graph
-> logits sampler
-> tokenizer decode
```

## Remaining Physical-Device Gap

```text
done: installed-root-only decode gate
done: clean skip when the app container is not staged
done: compiled .mlmodelc staging requirement identified
done: SE2 simulator installed-root decode from staged assets
pending: expose a named physical Watch SE2/SE3 destination in Xcode
pending: install the watch test app
pending: stage Qwen assets into the physical watch app container
pending: run installed-root decode on the device
pending: collect memory, jetsam, thermal, first-token, and decode tok/s data
```
