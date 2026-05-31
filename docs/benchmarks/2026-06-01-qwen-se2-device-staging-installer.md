# Qwen SE2 Device Staging Installer

Date: 2026-06-01
Branch: `codex/qwen-watch-se-runtime`
Target: physical Watch SE2 preparation

## Scope

This checkpoint turns the previous staging plan into an executable Swift
staging step. The new path can copy a manifest, shared stateful Core ML model,
and tokenizer into a target root that represents the watch app's Application
Support directory.

The physical device is still not connected in Xcode, so this run verifies the
installer with small same-layout fixtures and keeps the real Qwen 610 MB
artifact as a generated plan/hash input rather than duplicating it locally.

## Swift API

```text
ModelAssetStager.stage(plan:to:)
```

The stager:

```text
1. Creates the target root.
2. Copies each staging-plan item to its destination relative path.
3. Replaces stale target entries at the same path.
4. Recomputes SHA256 after copy.
5. Rejects expected/actual hash mismatches.
```

The staged target can then be opened as a normal `ModelAssetStore` root and
verified with the existing manifest/runtime checks.

## CLI

Plan-only:

```sh
swift run WatchLMBenchmark \
  --manifest tools/validation/fixtures/qwen3-0.6b-stateful-step-model-manifest.json \
  --asset-base artifacts/runtime-candidates \
  --device-profile watch-se-2 \
  --staging-plan \
  --output artifacts/benchmarks/qwen3-se2-device-staging-plan.json
```

Copy/install to a target root:

```sh
swift run WatchLMBenchmark \
  --manifest <manifest.json> \
  --asset-base <source-asset-root> \
  --device-profile watch-se-2 \
  --stage-to <watch-application-support-root> \
  --output <staging-result.json>
```

When a physical Watch SE2 is visible, `<watch-application-support-root>` should
be the installed watch app container's `Application Support/WatchLM` directory
or a host-side path that is then pushed into that container.

## Verification

```text
swift test --filter modelAssetStagerCopiesQwenPlanAndTargetStoreVerifiesInstalledState
swift test --filter runtimeBenchmarkCommandCanStageQwenAssetsToDestinationRoot
```

Both tests use the Qwen stateful-step manifest shape. The second test exercises
the CLI parse/command path and verifies the copied target root returns
`.installed` through `ModelAssetStore`.

## Current Interpretation

The physical-device gap is now narrower:

```text
done: generate device staging plan
done: copy plan items into a target root
done: verify target root through ModelAssetStore
pending: locate a named physical Watch destination
pending: map or copy into the actual watch app container
pending: run the existing stateful decode gate on physical SE2
```
