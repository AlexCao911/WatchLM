# Qwen Prefill Precision Isolation

## Purpose

Separate Qwen Core ML graph-correctness failures from quantization and stateful
KV failures. The earlier stateful-step int4/fp16 artifacts produced incorrect
logits. This test asks whether a simple logits-only Qwen prefill graph can match
PyTorch when precision is controlled.

## Method

All experiments used:

```text
model: Qwen/Qwen3-0.6B
graph: prefill
context: 16
prompt: Apple Watch local inference test.
validator: tools/validation/validate-coreml-prefill.py
compute units: CPU
topK: 10
```

The conversion script now accepts:

```text
--torch-dtype float16|float32|bfloat16|auto
```

This lets us separate model-load dtype from Core ML `compute_precision`.

## Results

| torch dtype | Core ML compute precision | max abs error | mean abs error | top-k agreement | top-1 |
| --- | --- | ---: | ---: | ---: | --- |
| float16 | float16 | 15.6758 | 3.3812 | 2/10 | mismatch |
| float32 | float16 | 15.6758 | 3.3812 | 2/10 | mismatch |
| float16 | float32 | 0.1973 | 0.0491 | 10/10 | match |
| float32 | float32 | 0.0221 | 0.0039 | 10/10 | match |

The logits-only `prefill-kv` fp16 diagnostic had the same bad profile as
logits-only `prefill` fp16, so the first-order failure is not caused by KV
outputs.

The traced PyTorch wrapper matched direct PyTorch exactly for both short and
long prompts:

```text
max_abs: 0.0
mean_abs: 0.0
top-k agreement: 10/10
```

## Interpretation

Qwen3-0.6B is convertible to a correct Core ML graph when Core ML keeps compute
precision at float32. The incorrect logits come from Core ML fp16
compute/lowering, not from:

```text
tokenizer mismatch
chat template mismatch
Swift sampler
PyTorch tracing
stateful KV alone
int4 compression alone
```

For the watch path this changes the optimization question. A global fp16 graph
is not a safe baseline for Qwen. The next useful experiments are mixed compute
precision policies that keep numerically sensitive operations or blocks in
float32 while still compressing storage aggressively enough for SE2/SE3.

## Next Step

Build a Qwen-specific mixed-precision conversion route:

```text
float32 compute baseline
-> identify fp16-sensitive subgraphs
-> keep sensitive attention/RoPE/softmax/norm regions high precision
-> compress weights with int4 only after fp16/fp32 mixed logits pass
```

Do not promote any Qwen int4 stateful artifact until the fp16/fp32 mixed
prefill gate reaches top-k agreement with the PyTorch teacher.
