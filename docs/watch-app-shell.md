# Watch App Shell Checkpoint

Date: 2026-05-29
Status: Implementation checkpoint

## Purpose

This checkpoint defines the first watchOS app shell before Xcode project generation. The shell must prove that the Apple Watch can run a self-contained local inference workflow around installed Core ML assets. It must not depend on iPhone, cloud, LAN, or server inference.

The current repository contains the pure Swift contracts the shell consumes:

- `ModelManifest`
- `DeviceProfile`
- `ContextVariantSelector`
- `ModelAssetState`
- `InferenceSessionState`
- `RuntimeTiming`
- `InferenceRuntime`
- `MockStreamingRuntime`

## Screens And States

### Missing Model

Swift state:

- `ModelAssetState.missing`
- `InferenceSessionState.idle`

Behavior:

- Show that the model is not installed.
- Offer install, developer sideload, or retry entry points.
- Do not offer remote inference fallback.

### Installing Model

Swift state:

- `ModelAssetState.installing(progress:)`
- `InferenceSessionState.idle`

Behavior:

- Show install progress.
- Keep generation disabled until the manifest and hashes validate.
- Allow cancellation of installation only if the asset manager can leave storage in a recoverable state.

### Ready

Swift state:

- `ModelAssetState.installed(manifest:)`
- `InferenceSessionState.idle`

Behavior:

- Select a context variant with `ContextVariantSelector`.
- Default SE 2 to 256 tokens and SE 3 to 512 tokens unless the installed manifest says otherwise.
- Enable short prompt entry and dictation text input.

### Generating

Swift state:

- `InferenceSessionState.prefill`
- `InferenceSessionState.decoding(generatedTokens:)`

Behavior:

- Stream tokens as they arrive from `InferenceRuntime`.
- Display elapsed phase timing from `RuntimeTiming` when available.
- Keep maximum response length short, matching the manifest and benchmark prompt envelope.
- Allow cancellation at the next token boundary.

### Cancelled

Swift state:

- `InferenceSessionState.cancelled`
- `InferenceRuntimeError.cancelled(partialTokens:)`

Behavior:

- Preserve already streamed partial text.
- Return to ready state without clearing installed model state.
- Do not restart generation automatically.

### Thermal Degraded

Swift state:

- `InferenceSessionState.thermalDegraded`

Behavior:

- Stop or prevent new generation while the watch is thermally constrained.
- Preserve current prompt text and model state.
- Resume only after the session controller allows another short foreground turn.

### Error Recovery

Swift state:

- `ModelAssetState.invalidHash(expected:actual:)`
- `ModelAssetState.incompatibleManifest(errors:)`
- `ModelAssetState.unavailableRuntime(reason:)`
- `InferenceSessionState.failed(message:)`
- `InferenceRuntimeError.modelAssetMissing`
- `InferenceRuntimeError.unavailableRuntime(reason:)`

Behavior:

- Distinguish asset errors from runtime errors.
- Offer reinstall for hash and manifest failures.
- Offer diagnostics for unavailable Core ML runtime.
- Do not switch to GGUF or remote inference as a product fallback.

## Type Mapping

| Shell responsibility | Swift contract |
| --- | --- |
| Identify SE 2 vs SE 3 | `DeviceProfile` |
| Decode installed model manifest | `ModelManifest` |
| Clamp context window | `ContextVariantSelector` |
| Represent model availability | `ModelAssetState` |
| Represent generation lifecycle | `InferenceSessionState` |
| Capture timings | `RuntimeTiming` |
| Abstract runtime invocation | `InferenceRuntime` |
| Verify UI without Core ML artifact | `MockStreamingRuntime` |

## Next Implementation Slice

The next app-facing slice should create a watchOS target or SwiftUI preview host that consumes `WatchLMCore` with `MockStreamingRuntime`. The first UI build should prove the state transitions above before adding a concrete Core ML adapter.
