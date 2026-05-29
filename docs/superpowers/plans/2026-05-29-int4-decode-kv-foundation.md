# Int4 Decode KV Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add the first real MiniCPM5 int4 compression path and generate Core ML prefill/decode graphs with explicit KV cache IO.

**Architecture:** Extend the existing conversion spike instead of creating a second tool. The script can now choose graph shape (`prefill`, `prefill-kv`, `decode`) and compression (`none`, `int8`, `int4`). KV cache construction stays outside the watch runtime for this spike, while Core ML graph IO is named exactly as the Swift runtime should consume later.

**Tech Stack:** Python, PyTorch, Transformers DynamicCache, coremltools ML Program conversion, Core ML palettization, Xcode `coremlc`.

---

## Task 1: Conversion CLI Surface

- [x] Add a failing Node test requiring `--compression none|int8|int4` and `--graph prefill|prefill-kv|decode`.
- [x] Add graph/compression arguments to `tools/conversion/convert-minicpm5-coreml.py`.
- [x] Preserve `--quantize` as a deprecated alias for `--compression int8`.
- [x] Run `node --test test/realConversionCli.test.js`.

## Task 2: Full-Model Int4 Spike

- [x] Implement Core ML int4 palettization through `OpPalettizerConfig(mode="kmeans", nbits=4)`.
- [x] Generate `artifacts/coreml/real-minicpm5-prefill-16-int4/prefill-16-int4.mlpackage`.
- [x] Compile it for watchOS 10 with `coremlc`.
- [x] Run PyTorch teacher vs Core ML logits validation.

Observed:

```text
int4 package bytes: 541164434
compiled size: about 516MB
top-1 match: false
top-10 agreement: 1/10
max absolute error: 15.26953125
```

Conclusion: full-model int4 is a valid size/compile proof, but it is not fidelity-safe. The next optimization pass should use mixed precision, not blanket int4.

## Task 3: Prefill KV Graph

- [x] Add `MiniCPMPrefillKVWrapper`.
- [x] Return `logits` plus `present_key_N` and `present_value_N` for all 24 layers.
- [x] Generate `artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage`.
- [x] Compile it for watchOS 10 with `coremlc`.

Observed:

```text
present_key/value shape: [1, 2, 16, 128]
package size: about 2.0GB
watchOS compile: succeeded
```

## Task 4: Decode KV Graph

- [x] Add `MiniCPMDecodeKVWrapper`.
- [x] Accept `token_id`, `position_id`, a decode causal mask, and `past_key/value_N` for all 24 layers.
- [x] Return `logits` plus one-token `new_key_N` and `new_value_N` for all 24 layers.
- [x] Generate `artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage`.
- [x] Compile it for watchOS 10 with `coremlc`.
- [x] Add `tools/validation/validate-coreml-decode.py`.
- [x] Run PyTorch teacher vs Core ML decode validation.

Observed:

```text
past_key/value shape: [1, 2, 16, 128]
new_key/value shape: [1, 2, 1, 128]
top-1 match: true
top-10 agreement: 9/10
logits max absolute error: 0.19140625
KV max absolute error: 0.087890625
```

## Task 5: Swift Runtime Graph Contract

- [x] Add manifest graph schema for `input_ids`, `position_ids`, `causal_mask`, `logits`, layered `present_key/value_N`, layered `past_key/value_N`, and layered `new_key/value_N`.
- [x] Validate the graph schema in both Node tooling and Swift manifest tests.
- [x] Add a manifest-driven `CoreMLPrefillDecodeBundle` initializer so Swift runtime consumes conversion output names instead of hard-coded assumptions.
- [x] Generate layered ML Program smoke graphs with rank-4 prefill/decode masks.
- [x] Run Swift end-to-end smoke inference through `Tokenizer -> PrefillGraph -> KVStore -> DecodeGraph -> LogitsSampler`.
- [x] Split Core ML runtime support into focused Swift files for graph bundle, feature IO, input state, logits sampler, and layered KV cache.

## Task 6: Swift KV Store and Logits Processor

- [x] Add `KVTensorLayout` so KV tensor shape, decode slice shape, scalar counts, and int8/fp16 memory budgets are explicit in Swift.
- [x] Replace the temporary layered cache with `CoreMLKVCacheStore`, including per-layer key/value shape checks and owned mutable buffers.
- [x] Keep the decode graph contract fixed to `[batch, kvHeads, context, headDim]` past tensors and `[batch, kvHeads, 1, headDim]` new-token slices.
- [x] Route Core ML logits through `CoreMLLogitsProcessor` into the existing `TokenLogit`/`GreedyTokenSampler` path instead of letting runtime assume a scalar `next_token`.
- [x] Add shared Swift logits policy for temperature, top-k, top-p, and repetition penalty.
- [x] Carry logits policy through `CoreMLPrefillDecodeBundle` and into the runtime decode loop.
- [x] Keep default greedy selection on a linear scan path so full-vocab decode does not require sorting every token step.

## Task 7: Stop Criteria and Result Metrics

- [x] Add generated token IDs to `InferenceResult` so benchmark and validation code can inspect token-level outputs.
- [x] Add explicit termination reasons for `maxTokens`, `endOfSequence`, and fixture/source exhaustion.
- [x] Use `DecodeStopCriteria` inside Core ML prefill/decode runtime instead of ad hoc loop checks.
- [x] Cover EOS-from-prefill behavior in Core ML smoke tests so first-token EOS does not enter decode.

## Task 8: Swift Active KV Window

- [x] Track `validTokenCount` and `activeTokenStartIndex` in `CoreMLKVCacheStore`.
- [x] Initialize KV cache validity from the real prompt token count after left-padding.
- [x] Move only active token slots while the prompt window still has left padding.
- [x] Expose `lastAppendMovedTokenCount` so benchmark code can report KV append copy pressure.
- [x] Cover active-window append behavior before the context window is full.

Observed:

```text
short prompt context=4 valid=2 first append moved 2 token slots
second append filled the window and moved 3 token slots
```

Conclusion: the explicit KV path now avoids full-window sliding during the early decode steps for short prompts. Once the context window is full it still performs a sliding copy, so a true ring-buffer or Core ML state/slice-view strategy remains the next memory optimization.

## Task 9: Swift Mixed Precision Policy Surface

- [x] Add a Swift `Quant` ownership boundary for mixed int4/int8/fp16 policy objects.
- [x] Parse manifest quantization strings into typed `QuantizedPrecision` values.
- [x] Reject uniform low-bit strategy, structural reduction, unsupported precisions, and non-int8 KV cache policy.
- [x] Protect the first and last two transformer layers by raising low-bit transformer components to at least int8.
- [x] Keep middle FFN layers eligible for int4 under the current fidelity-first baseline.

Observed:

```text
embedding: int8
lm_head: int8
norms: fp16
attention_qko: int8
attention_v: int8
middle ffn: int4
edge ffn: int8
kv_cache: int8
```

Conclusion: Swift now has a concrete model-optimization policy surface instead of treating quantization as opaque manifest strings. The conversion tool still needs to consume this policy for per-op Core ML compression.

## Task 10: Swift Decode Component Metrics

- [x] Add `InferenceMetrics` to `InferenceResult` without changing existing timing semantics.
- [x] Record prefill logits sampling time for the layered Core ML graph path.
- [x] Record per-step decode logits sampling time.
- [x] Record per-step KV append time.
- [x] Record per-step KV append moved token slots and scalar-copy counts.
- [x] Keep mock and legacy single-KV paths on empty default metrics.

Observed:

```text
context=4 prompt_tokens=2 decode KV moved token slots: [2, 3]
context=4 prompt_tokens=2 decode KV moved scalar counts: [4, 6]
```

Conclusion: the Swift runtime can now report which decode work comes from logits sampling versus explicit KV cache append/copy. This gives the next KV ring-buffer/stateful-cache optimization a direct before/after signal.

## Task 11: Swift MiniCPM Tokenizer Asset Path

- [x] Add a Swift `MiniCPMBytePairTokenizer` that reads Hugging Face `tokenizer.json`.
- [x] Parse BPE vocab, merge ranks, added tokens, and byte-level codec state in Swift.
- [x] Implement local encode with BOS policy, added-token splitting, ByteLevel pre-tokenization, and BPE merging.
- [x] Implement local decode for generated token IDs.
- [x] Compare Swift token IDs against the local `artifacts/hf/MiniCPM5-1B/tokenizer.json` smoke cases.

Observed:

```text
"Hello world!" -> [0, 36417, 1782, 22]
"Answer briefly." -> [0, 21742, 15020, 35]
"你好" -> [0, 75828]
rendered no-think chat prompt -> [0, 130072, 8448, 220, 19301, 130073, 220, 130072, 130071, 220, 8, 130063, 9, 130063]
```

Conclusion: the Swift runtime no longer needs a fixture-only tokenizer for the first real MiniCPM path. This is still a smoke-aligned tokenizer implementation; broader parity should add a larger prompt corpus against the Hugging Face tokenizer before physical-watch benchmarking.

## Task 12: Manifest-Selected Swift Runtime Assembly

- [x] Add `CoreMLRuntimeAssembler` so Swift can build runtime components from a `ModelManifest`, device profile, context request, and asset base URL.
- [x] Select SE2/SE3 artifact variants through the manifest instead of hand-wiring Core ML paths.
- [x] Verify prefill, decode, and tokenizer artifacts before assembly by default.
- [x] Load the selected `tokenizer.json` into `MiniCPMBytePairTokenizer`.
- [x] Build a manifest-schema-driven `CoreMLPrefillDecodeBundle`.
- [x] Return a `CoreMLPrefillDecodeRuntime` from the assembled bundle and tokenizer.
- [x] Extend artifact SHA256 to support deterministic `.mlpackage` directory digests.

Observed:

```text
manifest + watch-se-2 -> context 256
prefill path -> Models/MiniCPM5/prefill-256.mlpackage
decode path -> Models/MiniCPM5/decode-256.mlpackage
tokenizer path -> Models/MiniCPM5/tokenizer.json
verificationReport.isReady -> true
```

Conclusion: the Swift side now has an explicit route from installed artifacts to a runnable Core ML runtime. This still does not prove the real 256/512 MiniCPM Core ML artifacts fit or run on physical SE hardware; it proves the watch runtime can consume them once conversion produces verified assets.

## Task 13: Swift Seeded Sampling Path

- [x] Add a deterministic `SeededRandomNumberGenerator` for repeatable watch benchmark runs.
- [x] Add `SeededTokenSampler` so logits can be sampled probabilistically after temperature/top-k/top-p/repetition processing.
- [x] Add `TokenSamplingStrategy` with default greedy behavior and seeded stochastic behavior.
- [x] Let `CoreMLLogitsSampler` use an injected sampler instead of being hard-wired to greedy.
- [x] Carry sampling strategy through `CoreMLPrefillDecodeBundle` and `CoreMLRuntimeAssembler`.
- [x] Keep the layered Core ML runtime default on greedy while allowing seeded sampling for benchmark or product experiments.

Observed:

```text
same seed + same logits -> identical token sequence
flat logits + seeded sampling -> non-greedy token choices appear
manifest assembly can pass samplingStrategy through to the runtime bundle
```

Conclusion: the Swift chain now has a real `LogitsProcessor -> Sampler` boundary. This keeps deterministic greedy as the watch-safe default, while making stochastic sampling repeatable enough for quality and latency benchmarks.

## Task 14: Swift Runtime Streaming Events

- [x] Add `InferenceToken` with token index, optional token ID, decoded text, and first-token marker.
- [x] Add `InferenceStreamEvent.token` and `InferenceStreamEvent.completed`.
- [x] Add `StreamingInferenceRuntime` as the app-facing protocol for token-level incremental output.
- [x] Make `MockStreamingRuntime` emit token events before a final completion result.
- [x] Preserve cancellation behavior during streaming with already emitted partial tokens.
- [x] Make `CoreMLPrefillDecodeRuntime` stream tokens from the actual prefill/decode loops instead of waiting for the final result.

Observed:

```text
Mock stream -> token("local"), token(" answer"), completed("local answer")
Core ML layered smoke stream -> tokenID 5 "D", tokenID 6 "E", completed("DE")
stream cancellation after one token -> cancelled(partialTokens: ["one"])
```

Conclusion: the Swift chain now has a real `Streaming` boundary after sampling and tokenizer decode. Watch UI code can consume incremental tokens without depending on Core ML internals, and the same cancellation semantics apply to streaming and non-streaming generation.

## Task 15: Swift Runtime Benchmark Runner

- [x] Add `RuntimeBenchmarkPrompt` so Swift can run the same prompt envelope through real runtimes.
- [x] Add `RuntimeBenchmarkConfiguration` with source model, runtime, device profile, and context variant.
- [x] Add prompt-level benchmark results with text, token IDs, streamed token count, timing, metrics, termination reason, and error message.
- [x] Add aggregate summary with prompt counts, failures, total generated tokens, average first-token latency, and average decode speed.
- [x] Make `RuntimeBenchmarkRunner` prefer `StreamingInferenceRuntime` so benchmark runs exercise the app-facing incremental token path.
- [x] Preserve prompt-level failure reports when runtime load or generation fails.

Observed:

```text
Mock benchmark -> loadMs 3, prompt watch-001, text "AB", streamedTokenCount 2, decode 181.82 tok/s
load failure -> one prompt failure with "Model asset is not installed."
```

Conclusion: Swift now has an in-process benchmark runner that can produce comparable runtime evidence before and after quantization, KV-cache, sampler, or context-variant changes. It does not replace the host PyTorch/Core ML quality-drift tools; it gives the watch runtime its own stable measurement surface.

## Task 16: Swift Benchmark Telemetry

- [x] Add `RuntimeTelemetrySnapshot` with thermal state and resident memory.
- [x] Add `RuntimeTelemetryProbe` plus `ProcessRuntimeTelemetryProbe`.
- [x] Sample telemetry before load, after load, and after each prompt.
- [x] Report peak resident memory and thermal state sequence in `RuntimeBenchmarkReport` and `RuntimeBenchmarkSummary`.

Observed:

```text
telemetry snapshots -> nominal 90.0 MB, fair 110.5 MB, serious 105.0 MB
peakResidentMemoryMB -> 110.5
thermalStates -> nominal, fair, serious
```

Conclusion: Swift benchmark evidence now includes watch-relevant memory and thermal signals. This lets the SE2/SE3 path compare quantization, context variants, KV-cache layout changes, and sampling policies against more than token speed alone.

## Task 17: Core ML Graph IO Contract Validation

- [x] Add bundle-level validation for prefill/decode input and output feature names.
- [x] Validate layered logits/KV graphs across every expected layer.
- [x] Validate loaded `MLModelDescription` before storing Core ML models in the runtime.
- [x] Preserve graph mismatch diagnostics instead of wrapping them as generic load failures.

Observed:

```text
missing decode output -> decode outputs: new_value_1
runtime load mismatch -> decode outputs: new_key_1, new_value_1
CoreMLPrefillDecodeRuntimeTests -> 19 tests passed
```

Conclusion: Swift now actively checks that real Core ML artifacts expose the manifest-declared graph IO before inference starts. This reduces the risk that conversion silently changes logits/KV names while the watch runtime keeps driving a stale contract.

## Task 18: Core ML Graph Shape Contract Validation

- [x] Add bundle-level validation for prefill/decode input and output shapes.
- [x] Validate static MiniCPM prefill `input_ids`, `position_ids`, and `causal_mask` shapes.
- [x] Validate layered KV tensor shapes for `present`, `past`, and one-token `new` decode outputs.
- [x] Validate decode `token_id`, `position_id`, and `causal_mask` shapes.
- [x] Keep logits vocab dimension flexible while checking logits rank and batch.
- [x] Extract shapes from loaded `MLModelDescription` so runtime load validates real Core ML artifacts.

Observed:

```text
wrong decode new_key_0 -> shape [1, 2, 1, 64] expected [1, 2, 1, 128]
smoke Core ML modelDescription -> accepted layered graph shapes
CoreMLPrefillDecodeRuntimeTests -> 21 tests passed
```

Conclusion: Swift now rejects Core ML artifacts whose feature names match the manifest but whose tensor shapes would break prefill/decode or KV-cache updates. This is a necessary guard before moving SE2/SE3 from smoke graphs to real mixed int4/int8 MiniCPM artifacts.

## Task 19: Core ML Graph DType Contract Validation

- [x] Add bundle-level validation for prefill/decode input and output `MLMultiArray` data types.
- [x] Require Swift-generated token and position tensors to match `int32`.
- [x] Require masks and layered KV tensors to match `float16` for the current watch Core ML path.
- [x] Keep logits flexible across float16/float32/double while enforcing floating tensor types.
- [x] Extract dtypes from loaded `MLModelDescription` so runtime load validates real Core ML artifacts.

Observed:

```text
wrong input_ids dtype -> prefill inputs input_ids dtype float16 expected int32
smoke Core ML modelDescription -> accepted layered graph shapes and dtypes
CoreMLPrefillDecodeRuntimeTests -> 22 tests passed
```

Conclusion: Swift now rejects Core ML artifacts whose names and shapes match but whose tensor dtypes would mismatch the runtime-created arrays. This closes another load-time failure mode before moving from smoke graphs to real SE2/SE3 MiniCPM artifacts.

## Task 20: Swift KV Slot-Ring Update Strategy

- [x] Add `KVCacheUpdateStrategy` so Swift can choose between contiguous sliding and slot-ring KV updates.
- [x] Keep the legacy contiguous sliding path covered for graph/debug compatibility.
- [x] Add a slot-ring path that writes new decode K/V slices into free or oldest slots without moving existing past tensors.
- [x] Expose the last KV write slot so the decode input state can update its causal mask by slot occupancy instead of shifting the whole logical window.
- [x] Make the layered Core ML runtime default to slot-ring KV updates while preserving the fixed `[batch, kvHeads, context, headDim]` graph IO contract.

Observed:

```text
context=4 prompt_tokens=2 slot-ring writes -> slots [1, 0, 2]
slot-ring moved token slots -> [0, 0, 0]
layered Core ML smoke runtime -> generated D/E/F with KV append moved scalars [0, 0]
swift test -> 67 tests passed
```

Conclusion: The Swift decode path now has a real KV update optimization that does not require rebuilding the Core ML graph or copying the full past window every token. This is still explicit-buffer Core ML IO, not Core ML stateful cache, but it removes the largest avoidable Swift-side KV copy pressure before SE2/SE3 physical benchmarking.

## Task 21: KV Strategy Manifest and Benchmark Audit Trail

- [x] Validate `runtime.kvCacheMode` in Swift manifests and Node validation tooling.
- [x] Map `stateful-preferred` and `slot-ring` to Swift `KVCacheUpdateStrategy.slotRing`.
- [x] Map `contiguous-sliding` to Swift `KVCacheUpdateStrategy.contiguousSliding` for comparison runs.
- [x] Pass the manifest-selected KV update strategy through `CoreMLRuntimeAssembler` into `CoreMLPrefillDecodeBundle`.
- [x] Record `kvCacheUpdateStrategy` and per-step `kvAppendWriteIndices` in `InferenceMetrics`.
- [x] Preserve those metrics through `RuntimeBenchmarkRunner` so watch reports can say which KV strategy was measured.

Observed:

```text
manifest runtime.kvCacheMode=stateful-preferred -> bundle.kvCacheUpdateStrategy=slotRing
manifest runtime.kvCacheMode=contiguous-sliding -> bundle.kvCacheUpdateStrategy=contiguousSliding
layered Core ML smoke metrics -> kvCacheUpdateStrategy=slotRing, kvAppendWriteIndices=[1, 0]
Node manifest summary -> kvCacheMode=stateful-preferred
```

Conclusion: KV optimization is now an explicit runtime contract rather than a hidden Swift default. Real SE2/SE3 benchmark reports can distinguish slot-ring runs from contiguous-sliding fallback runs and compare copy pressure with `kvAppendMovedScalarCounts`.

## Task 22: Swift Benchmark Quality Drift Surface

- [x] Add `RuntimeQualityReference` so benchmark prompts can carry PyTorch teacher or host Core ML reference token IDs.
- [x] Add `RuntimeQualityDrift` to prompt results with reference source, compared token count, exact-match count, rounded token agreement, and first mismatch index.
- [x] Add `averageTokenAgreement` to benchmark summaries.
- [x] Compute quality drift inside `RuntimeBenchmarkRunner` from generated token IDs rather than relying on ad hoc logs.
- [x] Extend `MockStreamingRuntime` to emit generated token IDs so benchmark quality drift is covered through the same streaming path used by app-facing runtimes.

Observed:

```text
reference [10, 11] vs generated [10, 11] -> tokenAgreement=1.0, firstMismatch=nil
reference [10, 11, 12] vs generated [10, 99, 12] -> tokenAgreement=0.67, firstMismatch=1
RuntimeBenchmarkSummary.averageTokenAgreement -> 0.67 for the drift fixture
swift test -> 69 tests passed
```

Conclusion: Swift benchmark output can now carry quality evidence alongside latency, memory, thermal, sampling, and KV-copy metrics. Real PyTorch teacher or host Core ML validation tools can feed reference token IDs into the same prompt envelope, and SE2/SE3 watch runs can report quality drift without a separate side-channel.

## Task 23: Swift Benchmark Usability Gates

- [x] Add Swift `RuntimeBenchmarkGateTargets` with default SE2 and SE3 latency/decode targets matching the existing host-side benchmark contract.
- [x] Include optional quality agreement and peak memory thresholds so optimization passes can tighten gates as real artifacts mature.
- [x] Fail gates on critical thermal state by default.
- [x] Add `RuntimeBenchmarkGateResult` and `RuntimeBenchmarkGateMetrics` so reports can show measured values, targets, and explicit failure reasons.
- [x] Cover pass and multi-failure cases across latency, decode speed, quality agreement, resident memory, and thermal state.

Observed:

```text
SE3 defaults -> first token <= 3000ms, decode >= 3 tok/s, quality >= 0.8
SE2 defaults -> first token <= 5000ms, decode >= 1.5 tok/s, quality >= 0.7
failing fixture -> first-token, decode, quality, memory, and critical thermal failures reported together
swift test -> 71 tests passed
```

Conclusion: Swift can now judge whether a benchmark run is actually usable on the selected watch profile instead of only recording raw measurements. This makes later optimization passes more mechanical: run the artifact, compare gates, then decide whether to reduce context, adjust mixed precision, or optimize KV/state handling.

## Task 24: Conversion Mixed Precision Policy

- [x] Add a default `tools/conversion/mixed-precision-policy.json` that mirrors the Swift `MixedPrecisionPolicy` baseline.
- [x] Extend `convert-minicpm5-coreml.py` with `--compression mixed`, `--precision-policy`, and `--describe-compression-policy`.
- [x] Build a policy-derived compression plan that keeps protected edge-layer FFNs at int8 while allowing middle FFN projections to use int4.
- [x] Wire mixed compression into real Core ML compression as two selective passes: int8 linear quantization first, then int4 palettization for policy-selected ops.
- [x] Make op-name matching tolerate Core ML separator rewrites such as `mlp.down_proj` -> `mlp_down_proj`.
- [x] Record selector audit evidence by compression pass, component, layer, and sample op names.
- [x] Add `--source-mlpackage` so existing fp16 Core ML packages can be recompressed without re-running PyTorch tracing/conversion.
- [x] Add `layerOverrides` so a policy can restrict int4 to explicit transformer layers.
- [x] Cover the CLI, offline policy-plan contract, selector matching, selector audit, layer overrides, and source-package guard in `test/realConversionCli.test.js`.

Observed:

```text
mixed plan -> edge layer 0 ffn=int8, middle layer 12 ffn=int4
compression passes -> int8 linear_symmetric, then int4 kmeans_palettization
selector probe -> model_layers_12_mlp_down_proj matches mlp.down_proj
node --test test/realConversionCli.test.js -> 6 tests passed
```

Conclusion: the conversion path now consumes the same fidelity-first mixed precision policy language that Swift validates. The next artifact run can move from full-model int4/int8 spikes to mixed-compressed prefill/decode graphs, then use teacher-vs-runtime drift and SE2/SE3 benchmark gates to tune layer/component precision.

## Task 25: Real Context-16 Int8 and Mixed Artifact Evidence

- [x] Recompress existing real `decode-16.mlpackage` with `--compression mixed` and capture selector audit evidence.
- [x] Recompress existing real `prefill-kv-16.mlpackage` with `--compression mixed` and capture selector audit evidence.
- [x] Validate mixed decode logits and one-token KV against the PyTorch teacher.
- [x] Validate mixed prefill-kv logits against the PyTorch teacher.
- [x] Recompress existing real prefill-kv/decode graphs with global int8 to establish a quality baseline.
- [x] Validate int8 decode logits/KV and int8 prefill-kv logits against the PyTorch teacher.
- [x] Compile int8 prefill-kv and decode packages for watchOS with `coremlc`.

Observed:

```text
decode-16-mixed size: 870,136,324 bytes
prefill-kv-16-mixed size: 870,070,355 bytes
mixed selector audit: int8 selected 110 ops; int4 selected 60 FFN ops from layers 2...21
mixed decode validation: top1=false, top10 agreement=4/10, logits max error=9.556640625, KV max error=3.11376953125
mixed prefill-kv validation: top1=false, top10 agreement=2/10, logits max error=16.859375
decode-16-int8 size: 1,082,902,834 bytes
prefill-kv-16-int8 size: 1,082,836,862 bytes
int8 decode validation: top1=true, top10 agreement=9/10, logits max error=0.474609375, KV max error=0.1640625
int8 prefill-kv validation: top1=true, top10 agreement=10/10, logits max error=0.6669921875
watchOS coremlc: decode-16-int8 and prefill-kv-16-int8 compiled successfully
```

Conclusion: the first real explicit-KV MiniCPM5 chain has a viable context-16 int8 baseline for both prefill-kv and decode. The current mixed policy proves selective int4 compression works mechanically and reduces each graph to about 830MB, but it is too aggressive for fidelity. The next optimization pass should use the int8 artifacts as the teacher-preserving baseline, then try narrower FFN int4 windows before moving to context 256/512 or physical SE benchmarking.

## Task 26: Conservative FFN12 Mixed Policy Evidence

- [x] Add `tools/conversion/mixed-precision-policy-ffn12.json` with global int8 transformer weights and only layer 12 FFN set to int4.
- [x] Recompress real `decode-16.mlpackage` with the FFN12 policy.
- [x] Validate FFN12 decode logits and one-token KV against the PyTorch teacher.
- [x] Recompress real `prefill-kv-16.mlpackage` with the FFN12 policy.
- [x] Validate FFN12 prefill-kv logits against the PyTorch teacher.
- [x] Compile FFN12 prefill-kv and decode packages for watchOS with `coremlc`.

Observed:

```text
FFN12 selector audit: int8 selected 167 ops; int4 selected 3 FFN ops from layer 12
decode-16-mixed-ffn12 size: 1,072,265,112 bytes
prefill-kv-16-mixed-ffn12 size: 1,072,199,140 bytes
FFN12 decode validation: top1=true, top10 agreement=9/10, logits max error=0.890625, KV max error=0.3828125
FFN12 prefill-kv validation: top1=true, top10 agreement=10/10, logits max error=0.9697265625
watchOS coremlc: decode-16-mixed-ffn12 and prefill-kv-16-mixed-ffn12 compiled successfully
```

Conclusion: a narrow single-layer FFN int4 window preserves top-1 and top-10 agreement on the context-16 prompt while compiling for watchOS. Its size win over int8 is small, so the next quantization search should widen the layer override window gradually and plot size/error rather than jumping back to all middle FFN layers.

## Task 27: Swift Runtime Real Graph IO Alignment

- [x] Update the Swift layered-KV Core ML graph shape contract so prefill `input_ids` and `position_ids` are `[1, context]`, matching the real conversion outputs.
- [x] Update `CoreMLMiniCPMInputState` to produce batched `[1, context]` int32 arrays while preserving left padding, position ids, and causal mask behavior.
- [x] Regenerate layered Core ML smoke resources with the same prefill input rank used by the real MiniCPM graphs.
- [x] Add an optional macOS Swift integration test that loads local real int8 MiniCPM context-16 artifacts, uses the real MiniCPM tokenizer, runs prefill logits, builds the Swift KV cache, runs one decode step, samples from logits, and appends decode KV.
- [x] Add a macOS `.mlpackage` compile fallback for local tests while keeping watchOS aimed at precompiled `.mlmodelc` assets.

Observed:

```text
red test: CoreMLMiniCPMInputState produced input_ids shape [4], expected [1, 4]
red test: layered-KV bundle accepted vector prefill shapes [4]
real int8 Swift runtime test:
  prompt: Apple Watch local inference test.
  generated token ids: [242, 38]
  path: tokenizer -> prefill logits -> KVStore -> decode logits -> KV append
  duration: 42.373s on macOS host with local .mlpackage compile/load
swift test: 73 tests passed
node --test: 48 tests passed
python py_compile: conversion and validation scripts passed
git diff --check: passed
xcodebuild watchOS simulator SE 3: TEST SUCCEEDED
xcodebuild watchOS simulator SE 2: TEST SUCCEEDED, 70 Swift tests passed
```

Conclusion: the Swift side is no longer only smoke-graph infrastructure. It now consumes the same real MiniCPM graph IO rank and can execute a real context-16 int8 prefill/decode chain on the macOS host. This still is not a physical Apple Watch SE benchmark and does not prove context 256/512 memory fit, but it closes the prior gap between Python-produced logits/new_key/value graphs and Swift runtime expectations.

## Task 28: Swift Benchmark Artifact Provenance

- [x] Add `RuntimeBenchmarkArtifact` so Swift benchmark reports can carry quantization policy id, graph interface, prefill/decode/tokenizer paths, byte sizes, total size, and SHA-256 fields.
- [x] Add an optional artifact field to `RuntimeBenchmarkConfiguration` so every run can identify whether it measured int8, FFN12 mixed, or later mixed-window artifacts.
- [x] Add a manifest-selected artifact provenance test that computes directory/file byte sizes and carries manifest SHA-256 fields into the benchmark configuration.
- [x] Route the local real context-16 int8 Core ML test through `RuntimeBenchmarkRunner`, with a PyTorch-teacher token reference and artifact provenance attached.

Observed:

```text
artifact provenance fixture:
  prefill path: Models/MiniCPM5/prefill-256.mlpackage
  decode path: Models/MiniCPM5/decode-256.mlpackage
  tokenizer path: Models/MiniCPM5/tokenizer.json
  sizes: prefill=7 bytes, decode=6 bytes, tokenizer=9 bytes, total=22 bytes
real int8 benchmark runner test:
  configuration id: real-minicpm5-context16-int8-local
  quantization policy id: global-int8
  graph interface: logits-layered-kv
  reference source: pytorch-teacher-context16-int8-validation
  generated token ids: [242, 38]
  average token agreement: 1.0
  path: RuntimeBenchmarkRunner -> StreamingInferenceRuntime -> CoreMLPrefillDecodeRuntime
  duration: 36.198s on macOS host with local .mlpackage compile/load
```

Conclusion: Swift benchmark output can now distinguish artifact variants instead of only reporting runtime timings. This is necessary for the next optimization loop: run int8, FFN12 mixed, and wider mixed policies through the same Swift runner, then compare size, quality agreement, latency, memory, thermal state, and KV append behavior.

## Task 29: FFN10...13 Mixed Policy Evidence

- [x] Add `tools/conversion/mixed-precision-policy-ffn10-13.json` with global int8 transformer weights and layers 10, 11, 12, and 13 FFN set to int4.
- [x] Recompress real `decode-16.mlpackage` with the FFN10...13 policy and corrected `graph=decode` report metadata.
- [x] Recompress real `prefill-kv-16.mlpackage` with the FFN10...13 policy.
- [x] Validate FFN10...13 decode logits and one-token KV against the PyTorch teacher.
- [x] Validate FFN10...13 prefill-kv logits against the PyTorch teacher.
- [x] Compile FFN10...13 prefill-kv and decode packages for watchOS with `coremlc`.
- [x] Route FFN10...13 through the Swift `RuntimeBenchmarkRunner` local real-artifact test.

Observed:

```text
FFN10...13 selector audit: int8 selected 158 ops; int4 selected 12 FFN ops from layers 10...13
decode-16-mixed-ffn10-13 size: 1,040,350,041 bytes
prefill-kv-16-mixed-ffn10-13 size: 1,040,284,069 bytes
FFN10...13 decode validation: top1=true, top10 agreement=9/10, logits max error=1.51470947265625, KV max error=0.96533203125
FFN10...13 prefill-kv validation: top1=true, top10 agreement=9/10, logits max error=1.82421875
watchOS coremlc: decode-16-mixed-ffn10-13 and prefill-kv-16-mixed-ffn10-13 compiled successfully
Swift benchmark runner: generated token ids [242, 38], average token agreement 1.0, duration 42.230s on macOS host with local .mlpackage compile/load
```

Conclusion: FFN10...13 is the first wider mixed policy that still preserves context-16 top-1 on the local prompt while improving size over FFN12 by about 32MB per graph. It increases logits/KV drift and loses one top-10 prefill item, so it should not be promoted yet; it is a useful next point on the policy-size-quality frontier before trying FFN8...15 or moving to SE2/SE3 context sizes.

## Task 30: FFN8...15 Mixed Boundary Evidence

- [x] Add `tools/conversion/mixed-precision-policy-ffn8-15.json` with global int8 transformer weights and layers 8, 9, 10, 11, 12, 13, 14, and 15 FFN set to int4.
- [x] Recompress real `decode-16.mlpackage` with the FFN8...15 policy and corrected `graph=decode` report metadata.
- [x] Recompress real `prefill-kv-16.mlpackage` with the FFN8...15 policy.
- [x] Validate FFN8...15 decode logits and one-token KV against the PyTorch teacher.
- [x] Validate FFN8...15 prefill-kv logits against the PyTorch teacher.
- [x] Compile FFN8...15 prefill-kv and decode packages for watchOS with `coremlc`.
- [x] Route FFN8...15 through the Swift `RuntimeBenchmarkRunner` local real-artifact test.

Observed:

```text
FFN8...15 selector audit: int8 selected 146 ops; int4 selected 24 FFN ops from layers 8...15
decode-16-mixed-ffn8-15 size: 997,796,613 bytes
prefill-kv-16-mixed-ffn8-15 size: 997,730,641 bytes
FFN8...15 decode validation: top1=true, top10 agreement=8/10, logits max error=2.32843017578125, logits mean error=0.5262262225151062, KV max error=1.501708984375
FFN8...15 prefill-kv validation: top1=true, top10 agreement=9/10, logits max error=3.005859375, logits mean error=0.44225093722343445
watchOS coremlc: decode-16-mixed-ffn8-15 and prefill-kv-16-mixed-ffn8-15 compiled successfully
Swift benchmark runner: generated token ids [242, 38], average token agreement 1.0, duration 37.638s on macOS host with local .mlpackage compile/load
```

Conclusion: FFN8...15 is a mechanically viable and smaller boundary point, cutting each context-16 graph below 1GB while still preserving top-1 on the local prompt. It also shows the cost of widening the int4 window: decode top-10 agreement drops to 8/10 and KV/logits drift rises materially versus FFN10...13. Keep this result in the artifact frontier, but do not promote it as the default policy without broader prompt validation and SE2/SE3 device evidence.

## Task 31: Swift Benchmark Prompt Suite Loader

- [x] Add a Swift `RuntimeBenchmarkPromptSuite` that loads the shared `tools/benchmark/fixtures/benchmark-prompts.json` contract directly.
- [x] Extend `RuntimeBenchmarkPrompt` with `qualityChecks` so the Swift benchmark layer can carry the same qualitative rubric currently used by the host tooling.
- [x] Validate required prompt categories, duplicate/empty ids, prompt length, `maxNewTokens`, language, input, and non-empty quality checks in Swift.
- [x] Prove the shared fixture can feed `RuntimeBenchmarkRunner` without going through the JS validator.

Observed:

```text
red test: RuntimeBenchmarkPromptSuite did not exist and RuntimeBenchmarkPrompt could not carry qualityChecks
swift test --filter runtimeBenchmarkPromptSuite: 3 tests passed
shared fixture categories: zh_short_instruction, en_short_instruction, code_small_fix, watch_utility, safety_refusal
shared fixture prompt count: 10
```

Conclusion: benchmark prompts are no longer only a JS-side contract. Swift can now consume and validate the same prompt suite, which lets the real Core ML runner execute the exact optimization/evaluation prompts used by the host validation tooling.

## Task 32: Swift Teacher Reference Sidecar

- [x] Add `RuntimeBenchmarkPromptQualityReference` and `RuntimeBenchmarkQualityReferenceSuite` for prompt-id keyed teacher token references.
- [x] Add `RuntimeBenchmarkPromptSuite.applyingQualityReferences(...)` so Swift can merge PyTorch/Core ML host teacher tokens into the shared prompt suite.
- [x] Require non-empty sidecar source, unique prompt ids, non-empty token ids, and prompt-id existence checks.
- [x] Support a strict mode that fails when any benchmark prompt is missing a quality reference.
- [x] Prove the merged prompt suite flows through `RuntimeBenchmarkRunner` and produces `RuntimeQualityDrift` plus `averageTokenAgreement`.

Observed:

```text
red test: RuntimeBenchmarkQualityReferenceSuite did not exist and prompt suite had no applyingQualityReferences API
swift test --filter runtimeBenchmarkPromptSuite: 5 tests passed
teacher sidecar source: pytorch-teacher-minicpm5-context16
reference [10, 11, 12] vs generated [10, 11, 12] -> tokenAgreement=1.0
invalid sidecar errors include schema, source, prompt id, token ids, duplicate id, unknown id, and missing references
```

Conclusion: Swift benchmark inputs can now be evaluated against teacher-token references without relying on a side channel. The remaining work is to generate real PyTorch teacher token sidecars for the shared prompt suite, then run int8, FFN12, FFN10...13, and FFN8...15 through the same Swift runner.

## Task 33: PyTorch Teacher Reference Generator

- [x] Add `tools/benchmark/generate-teacher-references.py` so the shared prompt suite can produce Swift-readable prompt-id keyed teacher token sidecars.
- [x] Keep a dependency-light `--mock-token-ids` path for CI and schema checks without importing PyTorch or Transformers.
- [x] Add the real PyTorch teacher path that loads the local `artifacts/hf/MiniCPM5-1B` snapshot and emits greedy generated token IDs.
- [x] Verify the CLI with a Node smoke test and a one-prompt real MiniCPM5 teacher run.

Observed:

```text
red test: tools/benchmark/generate-teacher-references.py was missing
node --test test/teacherReferencesCli.test.js: 1 test passed
real teacher smoke: zh-short-001 -> token IDs [18487, 45105] with --max-new-tokens 2
sidecar path: artifacts/benchmarks/minicpm5-teacher-references-smoke.json
```

Conclusion: the Swift benchmark sidecar no longer has to be hand-authored. We now have a repeatable host tool that can generate teacher references for the shared prompt suite, which is the missing bridge before comparing int8, FFN12, FFN10...13, and FFN8...15 under one benchmark contract.

## Task 34: Swift Benchmark CLI

- [x] Add a Swift `WatchLMBenchmark` executable target plus `WatchLMBenchmarkSupport` command layer.
- [x] Load the shared prompt suite, optional teacher reference sidecar, prompt limit, and max-new-token cap from command-line arguments.
- [x] Run either a mock streaming runtime or a real Core ML MiniCPM explicit-KV runtime from the same command path.
- [x] Write `RuntimeBenchmarkReport` JSON with timing, token IDs, quality drift, telemetry, artifact size, and selected device/context metadata.
- [x] Verify the command layer through a Swift test and a real context-16 int8 Core ML CLI smoke run.

Observed:

```text
red test: WatchLMBenchmarkSupport target and RuntimeBenchmarkCommand did not exist
swift test --filter runtimeBenchmarkCommandMergesTeacherSidecarAndWritesMockReport: 1 test passed
swift run WatchLMBenchmark --runtime mock ...: prompts 1/1, avg_token_agreement 1.0
swift run WatchLMBenchmark --runtime coreml ...context16 int8...: prompts 1/1, avg_token_agreement 1.0
real int8 CLI smoke generated token IDs [18487, 45105] and text "限制回复"
real int8 CLI smoke loadMs 16326.848, firstTokenMs 10900.471, decode 0.11 tokens/s, peak resident memory 1710.89MB on macOS host
```

Conclusion: benchmark execution is now a first-class Swift path instead of being trapped inside tests. This makes the next optimization loop cleaner: generate a teacher sidecar, run each Core ML artifact policy through the same Swift CLI, then compare size, latency, memory, and token agreement before deciding which policy is worth taking to SE2/SE3 hardware.

## Task 35: Swift CLI Context-16 Artifact Matrix Smoke

- [x] Run the real Core ML Swift benchmark CLI against context-16 global int8.
- [x] Run the real Core ML Swift benchmark CLI against context-16 mixed FFN12.
- [x] Run the real Core ML Swift benchmark CLI against context-16 mixed FFN10...13.
- [x] Run the real Core ML Swift benchmark CLI against context-16 mixed FFN8...15.
- [x] Use the same PyTorch teacher sidecar, prompt limit, max-new-token cap, tokenizer, and benchmark report schema for all four runs.

Observed:

```text
policy         total artifact bytes  loadMs     firstTokenMs  decode tok/s  peak RSS MB  token agreement
int8           2,175,633,967         16326.848  10900.471     0.11          1710.89      1.0
FFN12          2,154,358,523         15409.565  10372.716     0.11          2725.45      1.0
FFN10...13     2,090,528,381         16196.665  10052.695     0.10          2829.34      1.0
FFN8...15      2,005,421,525         17058.381  10537.532     0.10          2817.06      1.0
generated IDs for all four policies: [18487, 45105]
generated text for all four policies: "限制回复"
```

Conclusion: the Swift CLI can now compare real Core ML artifact policies under one report contract. These are macOS host, context-16, one-prompt smoke numbers, so they are useful for verifying the benchmark plumbing and relative artifact metadata, not for claiming Apple Watch SE2/SE3 usable speed. The next quality step is broader teacher sidecars; the next deployment step is context 256/512 artifacts and physical/simulator SE profiling.

## Task 36: Full Prompt Suite Teacher Sidecar

- [x] Run `tools/benchmark/generate-teacher-references.py` against the full shared benchmark prompt suite.
- [x] Use the local `artifacts/hf/MiniCPM5-1B` PyTorch teacher snapshot without a max-new-token cap.
- [x] Emit the Swift-readable teacher sidecar to `artifacts/benchmarks/minicpm5-teacher-references-full.json`.
- [x] Summarize reference-token coverage before using it in Core ML runtime benchmarks.

Observed:

```text
source: pytorch-teacher-minicpm5
prompt count: 10
total reference tokens: 408
token lengths:
  zh-short-001: 48
  zh-short-002: 5
  en-short-001: 64
  en-short-002: 42
  code-fix-001: 48
  code-fix-002: 40
  watch-utility-001: 64
  watch-utility-002: 1
  safety-refusal-001: 48
  safety-refusal-002: 48
```

Conclusion: broad teacher-token coverage now exists locally for the shared Swift benchmark prompt suite. Running the entire sidecar through context-16 Core ML on the macOS host would be slow at the current observed decode speed, so the practical next step is a capped/batched Swift CLI run for policy comparison, then context 256/512 conversion once the artifact size strategy is selected.

## Task 37: Swift CLI Capped Teacher Reference Semantics

- [x] Reproduce the full-sidecar + `--prompt-limit 2` failure where strict sidecar validation rejected references for prompts outside the selected batch.
- [x] Add a regression test proving selected-prompt benchmarks can consume a full teacher sidecar.
- [x] Filter teacher references to the selected prompt ids after applying prompt limit.
- [x] Add a regression test proving `--max-new-tokens` capped benchmarks compare against the same capped teacher prefix.
- [x] Truncate selected teacher token references to each prompt's effective `maxNewTokens` before running the benchmark.
- [x] Re-run the real int8 context-16 batch2 CLI smoke with the full teacher sidecar.

Observed:

```text
before fix: prompt-limit 2 failed because references for en/code/watch/safety prompts were treated as unknown ids
after prompt filter: command ran, but averageTokenAgreement was 0.22 because 2 generated tokens were compared against 48/5-token references
after reference cap: swift run WatchLMBenchmark ...int8... --teacher minicpm5-teacher-references-full.json --prompt-limit 2 --max-new-tokens 2 -> prompts 2/2, avg_token_agreement 1.0
swift test --filter runtimeBenchmarkCommandMergesTeacherSidecarAndWritesMockReport: 1 test passed
```

Conclusion: the Swift CLI can now use one full teacher sidecar for batched/capped benchmark runs without distorting token agreement. This matters for slow host or watch runs because we can compare policies incrementally while preserving the same teacher corpus.

## Next Work

- Run int8, FFN12, FFN10...13, and FFN8...15 through capped or batched Swift prompt-suite benchmarks with the full teacher references.
- Validate slot-ring KV cache invariance against PyTorch/Core ML decode logits on real MiniCPM artifacts, then explore Core ML stateful cache or slice-view strategies if the graph/runtime supports them.
- Expand Swift tokenizer parity tests across Chinese, English, code, tool tags, and chat-template edge cases.
- Move from `context=16` to SE2 `context=256` and SE3 `context=512`.
