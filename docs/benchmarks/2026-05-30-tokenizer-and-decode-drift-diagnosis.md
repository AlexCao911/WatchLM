# Tokenizer And Decode Drift Diagnosis

## Scope

This note records the first diagnosis pass for category-balanced benchmark mismatches. It is separate from the policy matrix because it investigates why the baseline itself is not perfectly aligned.

## Question

After generating a context-16 teacher sidecar, the int8 host benchmark still had mismatches:

```text
int8 balanced5 averageTokenAgreement: 0.80
en-short-001 generated [416,826], teacher [416,4245]
watch-utility-001 generated [354,727], teacher [354,2305]
```

The first suspect was tokenizer or prompt preprocessing mismatch.

## Tokenizer Parity

Added a Swift parity test for every prompt in `tools/benchmark/fixtures/benchmark-prompts.json` against Hugging Face `AutoTokenizer(..., add_special_tokens=True)`.

```text
swift test --filter miniCPMBytePairTokenizerMatchesBenchmarkPromptSuiteHFReferences: 1 test passed
```

Conclusion: the Swift tokenizer matches the HF tokenizer for the full benchmark prompt suite. The current mismatches are not explained by tokenizer IDs.

## Split-Graph Teacher Check

Manual PyTorch split prefill/decode check with the same context-16 tensors produced:

```text
zh-short-001        [18487,45105]
en-short-001        [416,4245]
code-fix-001        [3342,801]
watch-utility-001   [354,2305]
safety-refusal-001  [1974,220]
```

These match the context-16 teacher sidecar prefixes.

## Decode Graph Validation

Core ML decode validation using PyTorch KV inputs:

```text
en-short-001:
  top1Matches: true
  topKAgreement: 10/10
  torchTopK: [4245,826,11420,5018,2793,4183,6971,2042,1903,3402]
  coremlTopK: [4245,826,11420,5018,2793,4183,6971,2042,1903,3402]

watch-utility-001:
  top1Matches: true
  topKAgreement: 10/10
  torchTopK: [2305,727,571,2077,901,558,2668,1017,1770,1758]
  coremlTopK: [2305,727,571,2077,901,558,2668,1017,1770,1758]
```

Conclusion: the standalone decode graph is aligned when fed PyTorch KV. The mismatch appears when the runtime feeds decode with Core ML prefill KV, or when host compute-unit behavior differs from CPU-only validation.

## Next

- Add an end-to-end Core ML prefill-KV-to-decode logits diagnostic for a selected prompt.
- Record prefill logits and decode logits top-k from the same Swift/Core ML execution path used by `WatchLMBenchmark`.
- Do not promote a mixed policy based only on aggregate token agreement until the baseline int8 mismatch source is understood.
