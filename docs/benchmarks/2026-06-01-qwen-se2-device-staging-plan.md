# Qwen SE2 Physical Device Staging Plan

Date: 2026-06-01
Branch: `codex/qwen-watch-se-runtime`
Target: Apple Watch SE 2 physical-device preparation

## Scope

This checkpoint adds the Swift-side staging contract required before a physical
SE2 run. The simulator could read artifacts directly from the Mac checkout, but
a physical Watch cannot. The device path needs an explicit Application Support
layout with the manifest, model artifact, and tokenizer copied into the watch
app container.

## Xcode Destination Check

Command:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
xcodebuild -showdestinations \
  -workspace .swiftpm/xcode/package.xcworkspace \
  -scheme WatchLM-Package
```

Result:

```text
Available watchOS destinations:
  Any watchOS Device
  Apple Watch SE (44mm) (2nd generation) simulator
  Apple Watch SE 3 simulators

No named physical Apple Watch destination was listed.
```

Interpretation: the local Xcode environment can run the simulator gates, but it
does not currently expose a connected physical SE2 destination for direct
`xcodebuild test`.

## Staging Plan Command

```sh
swift run WatchLMBenchmark \
  --manifest tools/validation/fixtures/qwen3-0.6b-stateful-step-model-manifest.json \
  --asset-base artifacts/runtime-candidates \
  --device-profile watch-se-2 \
  --staging-plan \
  --output artifacts/benchmarks/qwen3-se2-device-staging-plan.json
```

Result:

```text
items: 3
total_bytes: 609858910
destination: Application Support/WatchLM
```

## Required Device Layout

```text
Application Support/WatchLM/
  model-manifest.json
  Models/Qwen3/stateful-step-kv-256-fp32-compute-int8.mlpackage
  Models/Qwen3/tokenizer.json
```

The stateful-step Core ML artifact appears once in the staging plan and is
marked for both `prefill` and `decode`, which avoids double-copying the same
571 MB model.

## Hashes

```text
model actualSHA256:
eec61f0a0900c4cc66b10e7b82534a0cf9c2aa31845bf24baa483f12e7a84c03

tokenizer actualSHA256:
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
```

The Qwen stateful manifest fixture was updated to these real hashes so the
device-side `ModelAssetStore` verification path will not fail immediately with
a placeholder-hash mismatch.

## Current Interpretation

The next physical-device blocker is no longer an ambiguous runtime issue. The
remaining concrete steps are:

```text
1. Connect or expose a named physical Apple Watch SE 2 destination in Xcode.
2. Install a watchOS host/test app.
3. Copy the three staging-plan items into Application Support/WatchLM.
4. Run the same Qwen stateful decode gate on the device.
5. Record load, first-token latency, decode tok/s, memory or jetsam behavior,
   thermal state, and token agreement.
```
