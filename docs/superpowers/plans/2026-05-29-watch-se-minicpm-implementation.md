# Watch SE MiniCPM Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable foundation for Apple Watch SE-only MiniCPM5-1B local inference: auditable model manifests, benchmark prompt fixtures, benchmark report gates, and Swift runtime contracts that keep the project aligned with the fidelity-first Core ML architecture.

**Architecture:** Start with host-side evidence contracts in plain Node.js, then mirror the same contracts into a Swift package that the watchOS app shell can consume. The production runtime path remains Core ML `mlprogram` with split prefill/decode, fixed context variants, model assets outside the app bundle, and benchmark-controlled fallbacks.

**Tech Stack:** Node.js ESM with built-in `node:test` for host tooling; JSON fixtures for manifests, prompts, and reports; Swift Package Manager for watch runtime contracts; future watchOS SwiftUI target for the app shell.

---

## Source Inputs

- Design spec: `docs/superpowers/specs/2026-05-29-watch-se-minicpm-inference-design.md`
- ADR: `docs/architecture/0001-watch-se-minicpm-local-inference-decisions.md`
- Traceability matrix: `docs/requirements/watch-se-minicpm-traceability.md`
- User approval to proceed: conversation message on 2026-05-29, "开始执行 继续实现"

## Implementation Rules

- Preserve MiniCPM5-1B architecture and tokenizer as the first real artifact contract.
- Use Core ML `mlprogram` as the only production runtime contract.
- Treat GGUF and CPU-only execution as diagnostics only.
- Keep model assets outside the watchOS app bundle.
- Require benchmark evidence before reducing context, changing precision policy, optimizing `lm_head`, adding speculative decoding, vocabulary pruning, or layer pruning.
- Commit each completed part after its tests pass.

## Task 0: Commit the Implementation Plan

- [x] Add this plan at `docs/superpowers/plans/2026-05-29-watch-se-minicpm-implementation.md`.
- [x] Update `docs/requirements/watch-se-minicpm-traceability.md` so the approval gate reflects that execution is approved.
- [x] Review the plan for placeholder language and requirement coverage.
- [ ] Run:

```sh
git diff -- docs/superpowers/plans/2026-05-29-watch-se-minicpm-implementation.md docs/requirements/watch-se-minicpm-traceability.md
git status --short
```

- [ ] Commit:

```sh
git add docs/superpowers/plans/2026-05-29-watch-se-minicpm-implementation.md docs/requirements/watch-se-minicpm-traceability.md
git commit -m "docs: add Watch SE implementation plan"
```

## Task 1: Add Host Project Skeleton and Manifest Validator

Purpose: create the first executable contract for model assets and context selection.

Files:

- `package.json`
- `src/modelManifest.js`
- `test/modelManifest.test.js`
- `fixtures/sample-model-manifest.json`

Test-first steps:

- [x] Add `test/modelManifest.test.js` with coverage for:
  - valid MiniCPM5 manifest passes.
  - runtime must equal `coreml-mlprogram`.
  - source model must equal `openbmb/MiniCPM5-1B`.
  - architecture must preserve 24 layers, hidden size 1536, 16 query heads, 2 KV heads, and original tokenizer.
  - context variants must be a subset of `256`, `512`, and `1024`.
  - model bundle location must not be `app-bundle`.
  - quantization policy must include mixed precision and int8 KV cache.
  - `selectContextVariant` clamps requests to the largest supported variant that fits.
- [x] Run the failing test:

```sh
node --test test/modelManifest.test.js
```

Implementation steps:

- [x] Add `package.json` with `"type": "module"` and a `test` script that runs `node --test`.
- [x] Add `fixtures/sample-model-manifest.json` containing:
  - model id `openbmb/MiniCPM5-1B`
  - runtime `coreml-mlprogram`
  - device profiles `watch-se-2` and `watch-se-3`
  - context variants `256`, `512`, `1024`
  - architecture dimensions from the design spec
  - mixed precision quantization policy
  - asset storage location `application-support`
- [x] Add `src/modelManifest.js` exports:
  - `SUPPORTED_CONTEXT_VARIANTS`
  - `SUPPORTED_DEVICE_PROFILES`
  - `EXPECTED_MODEL_ID`
  - `EXPECTED_RUNTIME`
  - `EXPECTED_ARCHITECTURE`
  - `validateModelManifest(manifest)`
  - `assertValidModelManifest(manifest)`
  - `selectContextVariant(manifest, deviceProfile, requestedTokens)`
  - `summarizeModelManifest(manifest)`
- [x] Make validation return `{ ok, errors, warnings }` without throwing.
- [x] Make `assertValidModelManifest` throw one combined error with all validation messages.
- [x] Run:

```sh
node --test test/modelManifest.test.js
node --test
```

- [ ] Commit:

```sh
git add package.json src/modelManifest.js test/modelManifest.test.js fixtures/sample-model-manifest.json
git commit -m "feat: add model manifest validator"
```

## Task 2: Add Benchmark Prompt Fixtures

Purpose: keep quality checks tied to MiniCPM5 fidelity instead of only latency.

Files:

- `fixtures/benchmark-prompts.json`
- `src/benchmarkPrompts.js`
- `test/benchmarkPrompts.test.js`

Test-first steps:

- [x] Add tests that require prompt categories:
  - `zh_short_instruction`
  - `en_short_instruction`
  - `code_small_fix`
  - `watch_utility`
  - `safety_refusal`
- [x] Add tests that every prompt has `id`, `category`, `language`, `input`, `maxNewTokens`, and `qualityChecks`.
- [x] Add tests that `maxNewTokens` is between 16 and 96.
- [x] Add tests that prompt text length is compatible with a 256-token smoke baseline by using a conservative 4-characters-per-token estimate.
- [x] Run the failing test:

```sh
node --test test/benchmarkPrompts.test.js
```

Implementation steps:

- [x] Add `fixtures/benchmark-prompts.json` with at least two prompts per required category.
- [x] Add `src/benchmarkPrompts.js` exports:
  - `REQUIRED_PROMPT_CATEGORIES`
  - `loadBenchmarkPrompts(fileUrlOrPath)`
  - `validateBenchmarkPrompts(prompts)`
  - `groupPromptsByCategory(prompts)`
  - `estimatePromptTokens(prompt)`
- [x] Ensure validation reports all prompt errors in one result object.
- [x] Run:

```sh
node --test test/benchmarkPrompts.test.js
node --test
```

- [ ] Commit:

```sh
git add fixtures/benchmark-prompts.json src/benchmarkPrompts.js test/benchmarkPrompts.test.js
git commit -m "feat: add benchmark prompt fixtures"
```

## Task 3: Add Benchmark Report Schema and Gates

Purpose: make fallback decisions auditable before optimization begins.

Files:

- `fixtures/sample-benchmark-report.json`
- `src/benchmarkReport.js`
- `test/benchmarkReport.test.js`

Test-first steps:

- [x] Add tests that a report must include:
  - source model id
  - device profile
  - runtime
  - context variant
  - artifact size
  - load time
  - prefill latency
  - first token latency
  - decode tokens per second
  - peak resident memory
  - thermal state over five short turns
  - quality drift summary
  - fallback decision
- [x] Add SE 3 gate tests:
  - first visible token target is at most 3 seconds.
  - sustained decode target is at least 3 tokens per second.
- [x] Add SE 2 gate tests:
  - first visible token target is at most 5 seconds.
  - sustained decode target is at least 1.5 tokens per second.
- [x] Add tests that fallback decisions require evidence links or report sections.
- [x] Run the failing test:

```sh
node --test test/benchmarkReport.test.js
```

Implementation steps:

- [x] Add `fixtures/sample-benchmark-report.json` with one SE 2 diagnostic report and one SE 3 target report.
- [x] Add `src/benchmarkReport.js` exports:
  - `DEVICE_TARGETS`
  - `validateBenchmarkReport(report)`
  - `evaluateBenchmarkGates(report)`
  - `summarizeBenchmarkReport(report)`
  - `requiresFallbackEvidence(report)`
- [x] Ensure report validation uses the manifest constants from `src/modelManifest.js`.
- [x] Run:

```sh
node --test test/benchmarkReport.test.js
node --test
```

- [ ] Commit:

```sh
git add fixtures/sample-benchmark-report.json src/benchmarkReport.js test/benchmarkReport.test.js
git commit -m "feat: add benchmark report gates"
```

## Task 4: Add Local Validation CLI

Purpose: give future conversion and device benchmark artifacts a one-command contract check.

Files:

- `bin/watchlm-validate.js`
- `test/validationCli.test.js`
- `README.md`

Test-first steps:

- [x] Add CLI tests that run `node bin/watchlm-validate.js` against:
  - `fixtures/sample-model-manifest.json`
  - `fixtures/benchmark-prompts.json`
  - `fixtures/sample-benchmark-report.json`
- [x] Add a negative test fixture in the test file itself, written to a temporary directory during the test, so invalid runtime output exits non-zero.
- [x] Run the failing test:

```sh
node --test test/validationCli.test.js
```

Implementation steps:

- [x] Add `bin/watchlm-validate.js` with commands:
  - `manifest <path>`
  - `prompts <path>`
  - `report <path>`
  - `all --manifest <path> --prompts <path> --report <path>`
- [x] Add `README.md` with:
  - project purpose
  - local validation commands
  - first implementation status
  - note that real model artifacts are intentionally not committed
- [x] Run:

```sh
node --test test/validationCli.test.js
node --test
```

- [ ] Commit:

```sh
git add bin/watchlm-validate.js test/validationCli.test.js README.md
git commit -m "feat: add local validation CLI"
```

## Task 5: Add Swift Runtime Contract Package

Purpose: create watch-consumable runtime and asset state contracts without requiring a real Core ML artifact yet.

Files:

- `Package.swift`
- `.gitignore`
- `Sources/WatchLMCore/DeviceProfile.swift`
- `Sources/WatchLMCore/ModelManifest.swift`
- `Sources/WatchLMCore/ContextVariantSelector.swift`
- `Sources/WatchLMCore/ModelAssetState.swift`
- `Sources/WatchLMCore/InferenceSessionState.swift`
- `Tests/WatchLMCoreTests/ModelManifestTests.swift`
- `Tests/WatchLMCoreTests/ContextVariantSelectorTests.swift`
- `Tests/WatchLMCoreTests/ModelAssetStateTests.swift`
- `Tests/WatchLMCoreTests/InferenceSessionStateTests.swift`

Test-first steps:

- [x] Add Swift tests for manifest decoding from JSON matching `fixtures/sample-model-manifest.json`.
- [x] Add tests that context selection clamps to supported variants.
- [x] Add tests that asset states represent missing, installing, installed, invalid hash, incompatible manifest, and unavailable runtime.
- [x] Add tests that inference session state supports idle, prefill, decoding, cancelled, finished, failed, and thermal degraded states.
- [x] Run the failing tests:

```sh
swift test
```

Implementation steps:

- [x] Add `Package.swift` with library target `WatchLMCore`.
- [x] Implement Swift types with `Codable`, `Equatable`, and small pure functions only.
- [x] Avoid network, iPhone companion, and real Core ML calls in this package.
- [x] Run:

```sh
swift test
node --test
```

- [ ] Commit:

```sh
git add Package.swift Sources Tests
git commit -m "feat: add Swift runtime contracts"
```

## Task 6: Add Core ML Smoke Runtime Protocols

Purpose: make the eventual watchOS runtime measurable before MiniCPM artifact integration.

Files:

- `Sources/WatchLMCore/RuntimeTiming.swift`
- `Sources/WatchLMCore/InferenceRuntime.swift`
- `Sources/WatchLMCore/MockStreamingRuntime.swift`
- `Tests/WatchLMCoreTests/InferenceRuntimeTests.swift`

Test-first steps:

- [x] Add tests that the runtime protocol records load, prefill, first token, decode step, and total timings.
- [x] Add tests that mock streaming emits tokens incrementally.
- [x] Add tests that cancellation is observed at token boundaries.
- [x] Add tests that runtime errors are typed and user-visible.
- [x] Run the failing tests:

```sh
swift test
```

Implementation steps:

- [x] Implement protocols and mock runtime in pure Swift.
- [x] Keep Core ML imports out of `WatchLMCore`; a future watch target will provide the concrete adapter.
- [x] Run:

```sh
swift test
node --test
```

- [ ] Commit:

```sh
git add Sources/WatchLMCore Tests/WatchLMCoreTests
git commit -m "feat: add runtime timing contracts"
```

## Task 7: Add Conversion Artifact Contract

Purpose: prepare for Core ML conversion while keeping large generated models outside git.

Files:

- `conversion/README.md`
- `conversion/coreml-artifact-contract.json`
- `test/conversionContract.test.js`
- `.gitignore`

Test-first steps:

- [x] Add tests that conversion artifacts must declare:
  - source checkpoint id
  - source revision or checksum
  - tokenizer checksum
  - prefill model path
  - decode model path
  - context variant
  - quantization policy id
  - logits validation summary
  - excluded large artifact paths
- [x] Add tests that `.mlpackage`, `.mlmodelc`, `.gguf`, `.safetensors`, and generated benchmark outputs are ignored by default.
- [x] Run the failing test:

```sh
node --test test/conversionContract.test.js
```

Implementation steps:

- [x] Add `conversion/coreml-artifact-contract.json`.
- [x] Add `conversion/README.md` explaining reproducible artifact generation and why large model files stay outside git.
- [x] Add `.gitignore` entries for generated model and report artifacts.
- [x] Run:

```sh
node --test test/conversionContract.test.js
node --test
swift test
```

- [ ] Commit:

```sh
git add conversion .gitignore test/conversionContract.test.js
git commit -m "docs: add Core ML artifact contract"
```

## Task 8: Add Watch App Shell Plan Checkpoint

Purpose: create the next implementation checkpoint before Xcode project generation.

Files:

- `docs/watch-app-shell.md`
- `docs/requirements/watch-se-minicpm-traceability.md`

Steps:

- [x] Document the exact watchOS app shell screens:
  - missing model
  - installing model
  - ready
  - generating
  - cancelled
  - thermal degraded
  - error recovery
- [x] Document which `WatchLMCore` types each screen consumes.
- [x] Update traceability matrix evidence rows for completed host tooling and Swift runtime contracts.
- [x] Run:

```sh
node --test
swift test
git diff -- docs/watch-app-shell.md docs/requirements/watch-se-minicpm-traceability.md
```

- [ ] Commit:

```sh
git add docs/watch-app-shell.md docs/requirements/watch-se-minicpm-traceability.md
git commit -m "docs: define watch app shell checkpoint"
```

## Self-Review Checklist

- [x] Every user constraint maps to at least one implementation task.
- [x] The first executable tasks support Apple Watch-only local inference rather than phone, server, or LAN inference.
- [x] No task starts with layer pruning, hidden-size reduction, or a smaller replacement model.
- [x] Benchmark evidence exists before fallback paths are implemented.
- [x] Manifests and reports preserve auditability for model artifacts that cannot be committed.
- [x] Test commands are concrete and runnable from the repository root.
- [x] Commit commands are included after each completed part.
- [x] No placeholder markers remain in this plan.
