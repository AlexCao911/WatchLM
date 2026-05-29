# Core ML Smoke Inference Validation

Date: 2026-05-29
Host Xcode: Xcode 26.3, build 17C529
Simulator runtime: watchOS 26.2
Destination: Apple Watch SE 3 (44mm)
Branch: `codex/watch-se-minicpm-foundation`

## Scope

This run validates that `WatchLMCore` can load a compiled Core ML model and execute a real `MLModel.prediction` call inside both SwiftPM tests and the Apple Watch SE 3 simulator.

The model is a tiny identity smoke model, not MiniCPM5-1B. These timings prove the native Core ML inference path and test packaging only. They do not predict MiniCPM token speed, memory pressure, Neural Engine scheduling, or thermal behavior on physical Apple Watch SE hardware.

## Commands

```sh
swift test
node --test
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme WatchLM -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'
```

## Results

| Check | Result |
| --- | --- |
| macOS Swift tests | 15 passed, 0 failed |
| Node host tests | 37 passed, 0 failed |
| watchOS simulator Swift tests | 15 passed, 0 failed |
| watchOS simulator result bundle | `~/Library/Developer/Xcode/DerivedData/WatchLM-cidoceiepgeomqgqcsornmuwdlpw/Logs/Test/Test-WatchLM-2026.05.29_08-49-23-+0800.xcresult` |

macOS SwiftPM Core ML smoke print:

```text
WATCHLM_COREML_SMOKE output=7.5 load_ms=32.442 prediction_ms=0.474
```

Watch SE 3 simulator Core ML smoke print:

```text
WATCHLM_COREML_SMOKE output=7.5 load_ms=4.026 prediction_ms=0.237
```

Watch SE 3 simulator mock runtime print:

```text
WATCHLM_SIM_BENCH mock_short_turn iterations=1000 elapsed_ms=13.589 turns_per_second=73589.44
```

The watchOS simulator run also logged:

```text
[coreml] Failed to get the home directory when checking model path.
```

The test still passed, so this is recorded as a simulator/Core ML diagnostic warning rather than a functional failure.

## Findings

- `CoreMLSmokeRuntime` loads a compiled `.mlmodelc` bundle with `MLModelConfiguration.computeUnits = .all`.
- The runtime executes a real prediction through Core ML and returns the model output through the existing `InferenceResult` contract.
- Test resources now include separate compiled artifacts for macOS and watchOS so the same test can run under SwiftPM and the Apple Watch simulator.
- The next meaningful step is replacing the identity smoke model with MiniCPM-derived `prefill` and one-token `decode` Core ML artifacts while preserving this runtime contract.
