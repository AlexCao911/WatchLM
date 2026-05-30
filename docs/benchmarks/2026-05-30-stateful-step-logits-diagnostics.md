# Stateful Step Logits Diagnostics

Date: 2026-05-30

## Scope

This note records the first Swift-side top-k logits diagnostics for the
`stateful-step-kv` Core ML route.

It is separate from the quantization matrix. The matrix says which artifacts pass
or fail token agreement. This note explains where the first visible ranking drift
appears in the Swift inference chain.

## Swift Diagnostic Mode

`WatchLMBenchmark` now supports a diagnostics mode:

```text
swift run WatchLMBenchmark ... --diagnostics-top-k 5
```

The mode uses the same Swift tokenizer, Core ML graph interface selection, and
prompt loading as the benchmark runner, but writes top-k logits from:

- the final prompt step, reported as `prefillTopK`
- the first decode step, reported as `decodeTopK`

For `stateful-step-kv`, diagnostics load one shared Core ML model and reuse one
`MLState`, matching the runtime route.

## Artifacts

FP16 reference:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256/stateful-step-kv-256.mlmodelc
artifacts/benchmarks/stateful-step-kv-256-fp16-logits-diagnostics.json
```

Global int4:

```text
artifacts/coreml/compiled-macos-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlmodelc
artifacts/benchmarks/stateful-step-kv-256-int4-logits-diagnostics.json
```

Prompt:

```text
en-short-001
```

## Results

FP16:

```text
prefill top-5 token IDs: [1974, 591, 343, 416, 2452]
prefill top-1 margin:   0.4609375
decode top-5 token IDs: [10300, 220, 11439, 54, 282]
decode top-1 margin:   2.328125
```

Global int4:

```text
prefill top-5 token IDs: [5, 121400, 24, 26966, 8]
prefill top-1 margin:   11.2421875
decode top-5 token IDs: [67778, 21911, 7425, 6301, 84230]
decode top-1 margin:   1.1640625
```

## Interpretation

The global-int4 artifact has already diverged at the final prompt-step logits.
This means the quality failure is not merely a sampler issue and not only a
decode-loop accumulation issue.

The first bad generated token `[5]` follows directly from the int4 prefill logits,
while the fp16 graph chooses `[1974]`, matching the earlier teacher-aligned
stateful-step benchmark.

The next quantization work should therefore focus on the prompt-state-building
path first:

- compare fp16 vs compressed logits after shorter prompt prefixes
- isolate layer groups before the final prompt token
- avoid promoting global int4 for SE2 until top-k agreement recovers
