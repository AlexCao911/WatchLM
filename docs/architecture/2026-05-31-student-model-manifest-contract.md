# Student Model Manifest Contract

Date: 2026-05-31

## Purpose

The small-model pivot requires WatchLM to represent non-MiniCPM runtime
candidates without weakening the MiniCPM5-1B fidelity baseline.

Before this change, both Node validation and Swift `ModelManifest` validation
hard-coded:

```text
model.id:        openbmb/MiniCPM5-1B
layers:          24
hidden size:     1536
query heads:     16
KV heads:        2
head dimension:  128
context variants: 256, 512, 1024
```

That made a 125M/350M/600M student impossible to express as a first-class
runtime artifact.

## Contract

MiniCPM5-1B manifests still use strict baseline validation:

```text
model.id must be openbmb/MiniCPM5-1B
MiniCPM architecture dimensions must match the known baseline
tokenizer and vocabulary must be preserved
structuralReduction must be false
```

Non-MiniCPM manifests are allowed only when explicit:

```json
{
  "model": {
    "id": "watchlm/distilled-350m",
    "revision": "student-v0",
    "parameterCount": 350000000,
    "role": "runtime-candidate"
  }
}
```

For runtime candidates:

```text
runtime.graphSchema.layerCount must match architecture.layers
runtime.graphSchema.kvHeads must match architecture.kvHeads
runtime.graphSchema.headDimension must match architecture.headDimension
tokenizer.source and tokenizer.chatTemplate must be non-empty
tokenizer/vocabulary preservation may be false
structuralReduction may be true
context variant 128 is allowed
```

## Implemented Surfaces

Swift:

```text
Sources/ModelRuntime/Model/ModelManifest.swift
Tests/WatchLMCoreTests/ModelManifestTests.swift
```

Node validation:

```text
tools/validation/modelManifest.js
test/modelManifest.test.js
test/benchmarkReport.test.js
```

## Why This Matters

The Swift/Core ML inference chain is still the production path. This change only
allows the manifest layer to describe a smaller graph after we pick or distill
one.

It removes a structural blocker for the higher-leverage route:

```text
size gate -> student manifest -> Core ML conversion -> Swift benchmark
```

Without this contract, the sizing gate could say "350M is plausible" while the
runtime manifest system still rejected every non-MiniCPM artifact.

## Verification

Commands:

```bash
node --test
swift test
git diff --check
```

Results:

```text
Node:  107/107 passed
Swift: 121/121 passed
diff check: clean
```
