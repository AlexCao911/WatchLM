# Prefill Protected Batch Benchmark

## Scope

This note records the first Swift benchmark CLI run for:

```text
protected prefill-KV + int8 decode
```

It uses the same category-balanced context-16 prompt set as the earlier policy matrix. This is host-side evidence, not physical Apple Watch SE2/SE3 performance evidence.

## Command

```text
swift run WatchLMBenchmark --runtime coreml --prefill artifacts/coreml/real-minicpm5-prefill-kv-16-prefill-protected/prefill-kv-16-mixed.mlpackage --decode artifacts/coreml/real-minicpm5-decode-16-int8/decode-16-int8.mlpackage --tokenizer artifacts/hf/MiniCPM5-1B/tokenizer.json --teacher artifacts/benchmarks/minicpm5-teacher-references-context16-full.json --prompt-ids zh-short-001,en-short-001,code-fix-001,watch-utility-001,safety-refusal-001 --max-new-tokens 2 --context 16 --policy-id prefill-kv-fp16-attn-ffn12-int4-decode-int8 --id real-minicpm5-context16-prefill-protected-int8-decode-category-balanced --output artifacts/benchmarks/prefill-protected-int8-decode-category-balanced.json
```

## Summary

```text
prompts: 5/5
average token agreement: 0.80
average first token: 2117.59 ms
average decode: 93.55 tokens/sec
peak resident memory: 2469.48 MB
load: 15690.939 ms
total artifact bytes: 2334638889
```

## Prompt Results

```text
teacher prefixes:
  zh-short-001        [18487,45105]
  en-short-001        [416,4245]
  code-fix-001        [3342,801]
  watch-utility-001   [354,2305]
  safety-refusal-001  [1974,220]

protected prefill + int8 decode:
  zh-short-001        [18487,45105] agreement 1.0
  en-short-001        [416,4245]    agreement 1.0
  code-fix-001        [5028,58863]  agreement 0.0
  watch-utility-001   [354,2305]    agreement 1.0
  safety-refusal-001  [1974,220]    agreement 1.0
```

## Interpretation

The protected prefill policy fixed two previous int8 mismatches:

```text
en-short-001:      [416,826] -> [416,4245]
watch-utility-001: [354,727] -> [354,2305]
```

It also regressed the code prompt:

```text
code-fix-001: [3342,801] -> [5028,58863]
```

So the policy is a useful fidelity direction for prefill KV drift, but it is not yet a default promotion candidate. The next policy search should test whether the code regression comes from layer-12 FFN int4 in prefill or from the broader choice to keep attention fp16 while FFN remains int8.

## Next

Run two narrower prefill variants before moving to larger contexts:

```text
prefill attention fp16 + FFN int8 + no int4
prefill attention fp16 + FFN10...13 int4 only if the no-int4 variant keeps code-fix stable
```

Keep decode int8 during this search because protected prefill already restored `en-short-001` with int8 decode.
