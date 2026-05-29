# Real MiniCPM Core ML Conversion Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Convert a real `openbmb/MiniCPM5-1B` graph into Core ML as early as possible, so operator, memory, and package-size blockers are discovered before more watch-side plumbing is added.

**Architecture:** Keep full MiniCPM weights outside git under `artifacts/hf`. Add a reproducible conversion script under `tools/conversion` that can download model metadata/weights, trace a static prefill graph, convert to Core ML `mlprogram`, optionally quantize weights, and write a machine-readable conversion report.

**Tech Stack:** Python, PyTorch, Transformers, Hugging Face Hub, safetensors, coremltools 9, Xcode `coremlc`.

---

## Task 1: Prepare Environment And Metadata

- [x] Install conversion dependencies in `.venv`:

```sh
.venv/bin/python -m pip install torch transformers huggingface_hub safetensors tokenizers accelerate
```

- [x] Query Hugging Face metadata for `openbmb/MiniCPM5-1B`.

Observed:

```text
revision: 4e9de7a0778dc1c362e983e6858f0e77542cbdca
weight file: model-00000-of-00001.safetensors, 2161290912 bytes
architecture: LlamaForCausalLM, 24 layers, hidden 1536, 16 query heads, 2 KV heads
```

## Task 2: Add Conversion Script

Files:

- Create: `tools/conversion/convert-minicpm5-coreml.py`
- Modify: `.gitignore`

The script must:

- download the model snapshot into `artifacts/hf/MiniCPM5-1B`.
- load `AutoTokenizer` and `AutoModelForCausalLM`.
- trace a static short-context prefill wrapper using real MiniCPM weights.
- call `coremltools.convert(..., convert_to="mlprogram")`.
- optionally run Core ML weight quantization.
- save an `.mlpackage` and `conversion-report.json`.

Implementation detail added after first failures:

- pass explicit `position_ids` and a precomputed 4D additive causal mask into the traced prefill graph.
- force eager attention so the exported graph uses Core ML-friendly matmul/softmax operations.
- keep padding/mask construction outside the model graph; watchOS runtime must mirror this exactly.

## Task 3: Run First Real Conversion Attempt

Run:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-16 \
  --compute-precision float16
```

Expected result:

- If conversion succeeds, report the generated `.mlpackage` path and size.
- If conversion fails, preserve the failure in `conversion-report.json` with the exact stage and exception.

Observed attempts:

- Attempt 1 failed in Core ML conversion on `aten::diff` from Transformers packed-sequence mask detection.
- Attempt 2 removed `diff` by passing a 2D attention mask, then failed on `aten::new_ones` from dynamic causal mask construction.
- Attempt 3 moved `position_ids` and 4D causal mask outside the graph and succeeded.

Successful fp16 artifact:

```text
path: artifacts/coreml/real-minicpm5-prefill-16/prefill-16.mlpackage
bytes: 2161952910
```

Successful int8 artifact:

```text
path: artifacts/coreml/real-minicpm5-prefill-16-int8/prefill-16-int8.mlpackage
bytes: 1082818930
```

watchOS compiler check:

```text
command: xcrun coremlc compile ... --platform watchOS --deployment-target 10.0
output: artifacts/coreml/compiled-watchos-int8/prefill-16-int8.mlmodelc
compiled size: about 1.0GB
```

Xcode watch simulator build check:

```text
scheme: WatchLM
destination: platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)
result: BUILD SUCCEEDED
note: the Swift Package workspace exposes scheme WatchLM, not WatchLMCore
```

Logits validation against PyTorch teacher:

```text
fp16 max abs error: 0.216796875
fp16 mean abs error: 0.03411087393760681
fp16 top-10 agreement: 10/10
int8 max abs error: 0.6669921875
int8 mean abs error: 0.12139997631311417
int8 top-10 agreement: 10/10
top-1 match: true for both fp16 and int8
```

## Task 4: Decide Next Step From Evidence

- If short-context prefill converts, extend to SE2 prefill-256.
- If conversion fails on an operator, isolate the failing op and test a smaller wrapper.
- If conversion succeeds but package is too large, apply Core ML weight quantization and compare size.
- Only after prefill converts should decode/KV conversion be expanded.

Decision:

- Short-context prefill conversion is viable.
- int8 post-training Core ML weight quantization cuts size from about 2.16GB to about 1.08GB.
- The int8 package compiles for watchOS 10, but 1.0GB compiled size is too large to assume Watch SE2/SE3 runtime viability.
- Single-prompt logits validation is promising but not sufficient; the benchmark prompt suite still needs teacher/Core ML comparison.
- A monolithic full-model prefill package is still too large to assume Apple Watch SE2/SE3 viability.
- Next required conversion work is split decode with KV-cache IO and smaller fixed shapes, not more mock runtime code.
