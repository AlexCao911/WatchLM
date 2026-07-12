# Real Stateful KV Artifact Evidence

Date: 2026-05-30

## Scope

This note records the first real MiniCPM5-1B Core ML `stateful-kv` artifact
attempt.

It is separate from the conversion contract note. The contract note defines the
intended graph interface. This note records what happened when that contract was
used on real MiniCPM weights.

## Artifacts

Uncompressed stateful graph:

```text
artifacts/coreml/real-minicpm5-stateful-kv-16/stateful-kv-16.mlpackage
```

Int4 stateful graph:

```text
artifacts/coreml/real-minicpm5-stateful-kv-16-int4/stateful-kv-16-int4.mlpackage
```

Compiled variants:

```text
artifacts/coreml/compiled-watchos-stateful-kv-16-int4/stateful-kv-16-int4.mlmodelc
artifacts/coreml/compiled-macos-stateful-kv-16-int4/stateful-kv-16-int4.mlmodelc
artifacts/coreml/compiled-macos-stateful-kv-16/stateful-kv-16.mlmodelc
```

## Conversion Results

The real MiniCPM5-1B `stateful-kv` conversion completed for context 16:

```text
download_snapshot:          succeeded
load_tokenizer:             succeeded
load_model:                 succeeded
build_example_inputs:       succeeded
trace_stateful-kv:          succeeded
convert_stateful-kv_coreml: succeeded
```

The uncompressed package size was:

```text
2,162,303,710 bytes
```

The same graph compressed with global int4 palettization completed:

```text
compress_coreml_weights_int4: succeeded
```

The int4 package size was:

```text
541,513,520 bytes
```

This is the first real single-graph MiniCPM artifact at roughly the same disk
size as one previous explicit prefill/decode graph. The important size change is
that this shape does not require two 516 MB graph copies.

## Compile Results

The int4 graph passed watchOS 11 Core ML compilation:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile \
  artifacts/coreml/real-minicpm5-stateful-kv-16-int4/stateful-kv-16-int4.mlpackage \
  artifacts/coreml/compiled-watchos-stateful-kv-16-int4 \
  --platform watchOS \
  --deployment-target 11.0
```

The macOS 15 compiled int4 graph also completed.

Compiled sizes:

```text
watchOS int4 compiled: 516 MB
macOS int4 compiled:  516 MB
```

## Runtime Result

The compiled model is not yet executable through Core ML runtime on the host.

Swift benchmark command:

```text
swift run WatchLMBenchmark \
  --runtime coreml \
  --coreml-graph-interface stateful-kv \
  --prefill artifacts/coreml/compiled-macos-stateful-kv-16-int4/stateful-kv-16-int4.mlmodelc \
  --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json \
  --prompt-ids en-short-001 \
  --max-new-tokens 2 \
  --allow-missing-references \
  --context 16 \
  --policy-id stateful-kv-int4 \
  --id real-minicpm5-stateful-kv-16-int4-smoke \
  --output artifacts/benchmarks/stateful-kv-16-int4-smoke.json
```

Result:

```text
failedPromptCount: 1
peakResidentMemoryMB: 600.59
error: Failed to build the model execution plan using model.mil, Core ML error code -14.
```

A direct Swift `MLModel(contentsOf:)` check failed with the same `-14` error for
both the int4 compiled graph and the uncompressed compiled graph. Therefore this
is not caused only by int4 palettization.

## Current Diagnosis

The real graph MIL contains 48 state tensors and 48 `write_state` operations. The
state writes are not simple whole-tensor assignments. They are produced as
dynamic `slice_update` operations into rank-4 KV state tensors, followed by
`write_state`.

The existing smoke stateful model loads and runs, but it only writes one scalar
state with a simple whole-state update. That smoke test proves the Swift stateful
runtime API path, not that the current MiniCPM dynamic KV state update pattern is
accepted by the Core ML execution planner.

## Swift Runtime Changes From This Pass

The Swift side now has the pieces needed to keep testing this route without
hand-written runners:

- benchmark CLI option `--coreml-graph-interface stateful-kv`
- stateful Core ML bundle construction using one shared model URL
- stateful decode input names matching the real graph:
  `input_ids`, `position_ids`, `causal_mask`
- compact stateful prompt tensors so left-padding is not written into Core ML
  state
- reserved decode slots so a prompt can leave room for generated tokens inside
  the fixed KV state capacity

## Next Route

The next conversion attempt should avoid dynamic slice updates into state.

The most promising shape is a stateful single-token step graph:

```text
input_ids:     [1, 1]
position_ids:  [1, 1]
causal_mask:   [1, 1, 1, context + 1]
states:        past_key_N, past_value_N
output:        logits
state update:  full-state write after fixed sliding concat
```

Swift would build prompt KV by calling the same graph once per prompt token, then
continue decode with generated tokens. This trades first-token latency for a much
smaller single-model memory shape, which is likely the right tradeoff for Watch
SE2 if the execution planner accepts whole-state writes.
