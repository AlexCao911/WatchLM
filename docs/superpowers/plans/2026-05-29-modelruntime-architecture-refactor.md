# ModelRuntime Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Refactor WatchLM into a clearer ModelRuntime architecture that can grow from Core ML smoke inference into quantized MiniCPM5 prefill/decode on Apple Watch SE 2 and SE 3.

**Architecture:** Keep the public Swift product name `WatchLMCore` stable, but move its target path to `Sources/ModelRuntime` and organize files by runtime responsibility. Move host-side JavaScript tooling under `tools/conversion`, `tools/benchmark`, and `tools/validation` so conversion contracts, benchmark evidence, and validation code live near the domain they serve.

**Tech Stack:** Swift Package Manager, Swift Testing, Core ML, Node.js ESM, `node:test`, JSON validation fixtures.

---

## Architecture Assessment

The proposed structure is a good senior-level direction for this project:

```text
Sources/
  ModelRuntime/
    Core/
    Model/
    Tokenizer/
    Runtime/
      CoreML/
      Mock/
    Memory/
    Decode/
    Quant/
    Device/
    Eval/
    Security/
    Common/

tools/
  conversion/
  benchmark/
  validation/
```

It is clear enough for the current MiniCPM-on-watch goal if we apply two constraints:

- Do not create empty architectural buckets just to match the diagram. Materialize `Decode`, `Quant`, `Eval`, `Security`, and `Common` only when real code or contracts exist.
- Avoid `Common` as a dumping ground. Shared helpers should stay near their owning domain unless at least two concrete domains already depend on them.

For the current codebase, the immediate target layout is:

```text
Sources/ModelRuntime/
  Core/
    InferenceRuntime.swift
    InferenceSessionState.swift
    RuntimeTiming.swift
  Device/
    ContextVariantSelector.swift
    DeviceProfile.swift
  Memory/
    KVCache.swift
  Model/
    ModelAssetState.swift
    ModelManifest.swift
  Runtime/
    CoreML/
      CoreMLPrefillDecodeRuntime.swift
      CoreMLSmokeRuntime.swift
    Mock/
      MockStreamingRuntime.swift
  Tokenizer/
    Tokenizer.swift

tools/
  benchmark/
    benchmarkPrompts.js
    benchmarkReport.js
    fixtures/
      benchmark-prompts.json
      sample-benchmark-report.json
  conversion/
    README.md
    coreml-artifact-contract.json
    generate-coreml-smoke-model.py
  validation/
    modelManifest.js
    watchlm-validate.js
    fixtures/
      sample-model-manifest.json
```

## Task 1: Move Swift Runtime Into `Sources/ModelRuntime`

**Files:**
- Move: `Sources/WatchLMCore/InferenceRuntime.swift` -> `Sources/ModelRuntime/Core/InferenceRuntime.swift`
- Move: `Sources/WatchLMCore/RuntimeTiming.swift` -> `Sources/ModelRuntime/Core/RuntimeTiming.swift`
- Move: `Sources/WatchLMCore/InferenceSessionState.swift` -> `Sources/ModelRuntime/Core/InferenceSessionState.swift`
- Move: `Sources/WatchLMCore/DeviceProfile.swift` -> `Sources/ModelRuntime/Device/DeviceProfile.swift`
- Move: `Sources/WatchLMCore/ContextVariantSelector.swift` -> `Sources/ModelRuntime/Device/ContextVariantSelector.swift`
- Move: `Sources/WatchLMCore/KVCache.swift` -> `Sources/ModelRuntime/Memory/KVCache.swift`
- Move: `Sources/WatchLMCore/ModelManifest.swift` -> `Sources/ModelRuntime/Model/ModelManifest.swift`
- Move: `Sources/WatchLMCore/ModelAssetState.swift` -> `Sources/ModelRuntime/Model/ModelAssetState.swift`
- Move: `Sources/WatchLMCore/Tokenizer.swift` -> `Sources/ModelRuntime/Tokenizer/Tokenizer.swift`
- Move: `Sources/WatchLMCore/CoreMLSmokeRuntime.swift` -> `Sources/ModelRuntime/Runtime/CoreML/CoreMLSmokeRuntime.swift`
- Move: `Sources/WatchLMCore/CoreMLPrefillDecodeRuntime.swift` -> `Sources/ModelRuntime/Runtime/CoreML/CoreMLPrefillDecodeRuntime.swift`
- Move: `Sources/WatchLMCore/MockStreamingRuntime.swift` -> `Sources/ModelRuntime/Runtime/Mock/MockStreamingRuntime.swift`
- Modify: `Package.swift`

- [x] **Step 1: Point the existing target at the new target path**

```swift
.target(name: "WatchLMCore", path: "Sources/ModelRuntime"),
```

- [x] **Step 2: Move the files into the domain folders**

Use `git mv` for every path listed above so history remains readable.

- [x] **Step 3: Run Swift tests**

```sh
swift test
```

Expected: 20 tests pass.

## Task 2: Move Host Tooling Under `tools/`

**Files:**
- Move: `src/benchmarkPrompts.js` -> `tools/benchmark/benchmarkPrompts.js`
- Move: `src/benchmarkReport.js` -> `tools/benchmark/benchmarkReport.js`
- Move: `src/modelManifest.js` -> `tools/validation/modelManifest.js`
- Move: `bin/watchlm-validate.js` -> `tools/validation/watchlm-validate.js`
- Move: `fixtures/benchmark-prompts.json` -> `tools/benchmark/fixtures/benchmark-prompts.json`
- Move: `fixtures/sample-benchmark-report.json` -> `tools/benchmark/fixtures/sample-benchmark-report.json`
- Move: `fixtures/sample-model-manifest.json` -> `tools/validation/fixtures/sample-model-manifest.json`
- Move: `conversion/README.md` -> `tools/conversion/README.md`
- Move: `conversion/coreml-artifact-contract.json` -> `tools/conversion/coreml-artifact-contract.json`
- Move: `scripts/generate-coreml-smoke-model.py` -> `tools/conversion/generate-coreml-smoke-model.py`
- Modify: `tools/validation/watchlm-validate.js`
- Modify: `test/*.test.js`
- Modify: `README.md`

- [x] **Step 1: Move files into domain-specific tool folders**

Use `git mv` for all paths listed above.

- [x] **Step 2: Update imports**

`tools/validation/watchlm-validate.js` should import:

```js
import { loadBenchmarkPrompts } from "../benchmark/benchmarkPrompts.js";
import { summarizeBenchmarkReport, validateBenchmarkReport } from "../benchmark/benchmarkReport.js";
import { assertValidModelManifest, summarizeModelManifest } from "./modelManifest.js";
```

- [x] **Step 3: Update tests and README paths**

Use these canonical fixture and CLI paths:

```text
tools/validation/watchlm-validate.js
tools/validation/fixtures/sample-model-manifest.json
tools/benchmark/fixtures/benchmark-prompts.json
tools/benchmark/fixtures/sample-benchmark-report.json
tools/conversion/coreml-artifact-contract.json
tools/conversion/generate-coreml-smoke-model.py
```

- [x] **Step 4: Run Node tests**

```sh
node --test
```

Expected: 40 tests pass.

## Task 3: Update Core ML Smoke Generation Paths

**Files:**
- Modify: `tools/conversion/generate-coreml-smoke-model.py`

- [x] **Step 1: Make the script resolve the repository root from `tools/conversion`**

```python
ROOT = Path(__file__).resolve().parents[2]
```

- [x] **Step 2: Regenerate smoke models**

```sh
.venv/bin/python tools/conversion/generate-coreml-smoke-model.py
```

Expected: macOS and watchOS identity, prefill, and decode `.mlmodelc` resources are regenerated under `Tests/WatchLMCoreTests/Resources`.

## Task 4: Verify Watch Runtime After Refactor

**Files:**
- No new files.

- [x] **Step 1: Run SwiftPM tests**

```sh
swift test
```

Expected: 20 tests pass.

- [x] **Step 2: Run Apple Watch SE 3 simulator tests**

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme WatchLM -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'
```

Expected: 20 tests pass and `WATCHLM_PREFILL_DECODE_SMOKE` is printed.

## Task 5: Record Architecture Outcome

**Files:**
- Create: `docs/architecture/0002-modelruntime-source-layout.md`

- [x] **Step 1: Record the architecture decision**

The ADR must state:

- The proposed layout is accepted with constraints.
- `WatchLMCore` remains the Swift product for API stability.
- `Sources/ModelRuntime` becomes the implementation root.
- `tools/` owns conversion, benchmark, and validation workflows.
- Empty domains are deferred until real code exists.

- [x] **Step 2: Run final validation**

```sh
git diff --check
node --test
swift test
```

Expected: all pass.

- [x] **Step 3: Commit**

```sh
git add .
git commit -m "refactor: clarify model runtime architecture"
```
