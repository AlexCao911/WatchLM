# SE2 Real Device Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add the remaining runtime foundations needed before a real quantized MiniCPM5 artifact can be installed and benchmarked on Apple Watch SE 2.

**Architecture:** Keep heavy model conversion outside git, but make the watch-side runtime strict about artifact paths, per-file hashes, tokenizer presence, and decode stop behavior. Add concrete `Security` and `Decode` domains now because they have real source ownership.

**Tech Stack:** Swift Package Manager, Swift Testing, CryptoKit, Foundation, Core ML smoke tests, Node validation tooling.

---

## Scope

This plan does not download or commit MiniCPM weights. It adds the code needed to accept real SE2 artifacts once conversion produces them:

- independent prefill/decode/tokenizer paths and SHA256 digests.
- SHA256 artifact verification usable on watchOS.
- readiness reports for missing, verified, and mismatched model files.
- greedy decode selection and EOS/max-new-token stop criteria.
- `.swiftpm/` ignored as generated Xcode workspace metadata.

## Tasks

- [x] **Task 1: Write failing Swift tests**

Files:

- `Tests/WatchLMCoreTests/ModelManifestTests.swift`
- `Tests/WatchLMCoreTests/ArtifactSecurityTests.swift`
- `Tests/WatchLMCoreTests/DecodePolicyTests.swift`

Expected red failure:

```text
cannot find 'ArtifactDigest' in scope
cannot find 'ModelArtifactVerifier' in scope
cannot find 'GreedyTokenSampler' in scope
SelectedModelArtifact has no member 'prefillSHA256'
```

- [x] **Task 2: Implement manifest digest fields**

Files:

- `Sources/ModelRuntime/Model/ModelManifest.swift`
- `Tests/WatchLMCoreTests/TestSupport.swift`
- `tools/validation/fixtures/sample-model-manifest.json`
- `tools/validation/modelManifest.js`
- `test/modelManifest.test.js`

Add optional `tokenizerPath`, `prefillSHA256`, `decodeSHA256`, and `tokenizerSHA256` fields to model artifact variants and selected artifacts.

- [x] **Task 3: Implement artifact security verification**

Files:

- `Sources/ModelRuntime/Security/ArtifactDigest.swift`
- `Sources/ModelRuntime/Security/ModelArtifactVerifier.swift`

Use CryptoKit SHA256 and report per-file readiness for prefill, decode, and tokenizer files.

- [x] **Task 4: Implement decode policy**

Files:

- `Sources/ModelRuntime/Decode/DecodePolicy.swift`

Add `TokenLogit`, `GreedyTokenSampler`, and `DecodeStopCriteria`.

- [x] **Task 5: Verify and commit**

Run:

```sh
git diff --check
node --test
swift test
```

Expected: all pass.
