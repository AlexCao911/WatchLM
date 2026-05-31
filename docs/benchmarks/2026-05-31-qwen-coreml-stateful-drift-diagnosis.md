# Qwen Core ML Stateful Drift Diagnosis

## Purpose

Record the current blocker on the Qwen Apple Watch path after switching from
MiniCPM5-1B to `Qwen/Qwen3-0.6B`.

The Swift/CoreML inference chain now reaches real logits and sampling, but the
active Qwen stateful Core ML graph does not agree with the PyTorch teacher.
This means the next work item is graph correctness, not another blind
quantization sweep.

## Current Swift Path

The active host path is:

```text
Qwen3 chat template
-> tokenizer with add_bos_token=false
-> Core ML stateful-step-kv graph
-> logits top-k / sampler
-> token decode
```

The benchmark CLI now supports the Qwen-specific options needed for this path:

```text
--coreml-layer-count 28
--coreml-kv-heads 8
--coreml-head-dim 128
--coreml-compute-units all|cpu-only|cpu-and-gpu|cpu-and-neural-engine
--tokenizer-add-bos true|false
--tokenizer-eos-token-ids 151645
--chat-template qwen3-nonthinking
```

The tokenizer/template parity smoke uses the local Hugging Face tokenizer and
matches the expected Qwen3 non-thinking prompt token IDs.

## Int4 Runtime Smoke

Artifact:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlpackage
```

Command shape:

```sh
swift run WatchLMBenchmark \
  --runtime coreml \
  --prefill artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-int4/stateful-step-kv-256-int4.mlpackage \
  --tokenizer artifacts/hf/Qwen3-0.6B/tokenizer.json \
  --coreml-graph-interface stateful-step-kv \
  --coreml-layer-count 28 \
  --coreml-kv-heads 8 \
  --coreml-head-dim 128 \
  --tokenizer-add-bos false \
  --tokenizer-eos-token-ids 151645 \
  --chat-template qwen3-nonthinking \
  --context 256 \
  --device-profile watch-se-2 \
  --source-model Qwen/Qwen3-0.6B \
  --policy-id qwen3-0.6b-stateful-step-kv-256-int4 \
  --id qwen3-0.6b-qwen-template-smoke \
  --prompt-ids en-short-001 \
  --max-new-tokens 8 \
  --allow-missing-references \
  --output artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-int4-qwen-template-smoke.json
```

Observed:

```text
prompt success: 1/1
artifact + tokenizer total: 310,753,464 bytes
host load: 8,227.251 ms
first token / prefill: 15,995.745 ms
decode throughput: 2.28 tokens/sec
host peak RSS: 608.2 MB
generated token IDs: [661, 26982, 43328, 10291, 61898, 38510, 50944, 31976]
generated text: "udPlainpcionottenquoiIVERSitantpoon"
```

This proves the Swift/CoreML runtime path executes real Qwen inference. It also
shows the output quality is broken before any watch deployment claim can be
made.

## Teacher Baseline

For prompt fixture `en-short-001`, after applying the Qwen3 non-thinking chat
template, the prompt token count is 30:

```text
[151644, 872, 198, 840, 20772, 304, 825, 2805, 14311, 3170,
 264, 6718, 855, 7559, 14, 18196, 4771, 8609, 3736, 44378,
 13, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
```

PyTorch teacher greedy decode for eight tokens:

```text
token IDs: [32, 6718, 855, 7559, 14, 18196, 4771, 8609]
text: "A split prefill/decode graph helps"
```

Teacher prefill top tokens start with:

```text
[(32, 31.203125), (20193, 26.59375), (785, 22.75), (641, 21.109375)]
```

## Drift Evidence

The int4 Core ML graph does not have teacher overlap at top-1 for the same
prompt. Its prefill top tokens start with:

```text
[(661, 15.3359375), (15884, 12.9453125), (13888, 12.484375)]
```

The fp16 Core ML graph is also wrong, so this is not currently an int4
quantization issue. Its prefill top tokens start with:

```text
[(103032, 10.6328125), (99159, 10.578125), (54599, 10.421875)]
```

Running fp16 with `--coreml-compute-units cpu-only` still disagrees with the
teacher and produces a different wrong distribution. This makes an ANE/GPU-only
backend issue unlikely.

A prefix sweep shows the stateful-step graph is wrong from very short prefixes,
not only after long KV accumulation. The prefix-1 and prefix-16 fp16 CPU-only
results even returned all-zero top logits in the diagnostic report.

Changing the blocked attention-mask value from fp16 minimum to `-10000` did not
change the prefix diagnostic result.

## Conversion Isolation

The PyTorch `MiniCPMStatefulStepKVWrapper` reused for Qwen conversion matches
the teacher when run directly. A `torch.jit.trace` version of the same wrapper
also matches the teacher.

That narrows the likely fault to the Core ML conversion/runtime representation,
especially the stateful KV update pattern, state layout, or MIL lowering, not
to tokenizer, sampler, chat template, or the PyTorch wrapper logic.

An attempted `stateful-kv` fp16 Qwen graph converted but could not be used by
the current Swift diagnostics path, and the generation path failed to build the
Core ML execution plan with error `-14`. That route remains useful as a design
reference, but it is not an immediate replacement yet.

## Current Interpretation

The blocker is:

```text
Core ML stateful graph correctness for Qwen3-0.6B
```

It is not:

```text
GGUF integration
missing Swift runtime plumbing
sampler-only behavior
global int4 quality loss
```

The next useful step is to add a hard teacher-vs-CoreML top-k gate to conversion
validation and then isolate graph correctness with a smaller explicit-KV or
non-stateful sanity graph before resuming int4/watch SE deployment tuning.
