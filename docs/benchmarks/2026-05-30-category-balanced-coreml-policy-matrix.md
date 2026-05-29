# Category-Balanced Core ML Policy Matrix

## Scope

This note records a small category-balanced host benchmark matrix for the existing context-16 Core ML artifacts.

This is not Apple Watch SE2/SE3 performance evidence. It is a host-side sanity check for benchmark plumbing, context-aligned teacher references, and prompt-sensitive quantization drift.

## Setup

```text
runtime: WatchLMBenchmark Swift CLI
teacher sidecar: artifacts/benchmarks/minicpm5-teacher-references-context16-full.json
context: 16
device profile metadata: watch-se-2
max new tokens: 2
prompt ids: zh-short-001,en-short-001,code-fix-001,watch-utility-001,safety-refusal-001
```

## Results

```text
policy       avg agreement  avg firstTokenMs  avg decode tok/s  peak RSS MB
int8         0.80           2178.89           80.71             2550.44
FFN12        0.70           2200.29           79.84             2851.33
FFN10...13   0.90           2217.75           77.35             2708.61
FFN8...15    0.50           2224.10           77.29             3045.23
```

## Prompt-Level Agreement

```text
teacher prefixes:
  zh-short-001        [18487,45105]
  en-short-001        [416,4245]
  code-fix-001        [3342,801]
  watch-utility-001   [354,2305]
  safety-refusal-001  [1974,220]

int8:
  zh-short-001        [18487,45105] agreement 1.0
  en-short-001        [416,826]     agreement 0.5
  code-fix-001        [3342,801]    agreement 1.0
  watch-utility-001   [354,727]     agreement 0.5
  safety-refusal-001  [1974,220]    agreement 1.0

FFN12:
  zh-short-001        [18487,45105] agreement 1.0
  en-short-001        [416,4245]    agreement 1.0
  code-fix-001        [5028,58863]  agreement 0.0
  watch-utility-001   [354,727]     agreement 0.5
  safety-refusal-001  [1974,220]    agreement 1.0

FFN10...13:
  zh-short-001        [18487,45105] agreement 1.0
  en-short-001        [416,4245]    agreement 1.0
  code-fix-001        [3342,801]    agreement 1.0
  watch-utility-001   [354,727]     agreement 0.5
  safety-refusal-001  [1974,220]    agreement 1.0

FFN8...15:
  zh-short-001        [18487,45105] agreement 1.0
  en-short-001        [242,416]     agreement 0.0
  code-fix-001        [2533,220]    agreement 0.0
  watch-utility-001   [354,727]     agreement 0.5
  safety-refusal-001  [1974,220]    agreement 1.0
```

## Interpretation

- Context alignment improved the validity of the comparison, but agreement is still not perfect even for int8.
- The remaining int8 mismatches should be investigated before treating mixed-policy differences as pure quantization drift.
- Likely suspects are Swift tokenizer parity on broader prompts, Core ML/PyTorch logits drift around close candidates, and exact prompt preprocessing differences.
- FFN8...15 remains too unstable for promotion.
- FFN10...13 looks best on this tiny batch, but this is not enough to promote it because earlier one-prompt/top-k validation showed higher KV/logits drift than narrower policies.

## Next

Run tokenizer parity and PyTorch-vs-CoreML top-k checks on the mismatching prompts before spending time on larger context conversion.
