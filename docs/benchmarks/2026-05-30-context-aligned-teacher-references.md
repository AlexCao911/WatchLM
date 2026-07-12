# Context-Aligned Teacher References

## Scope

This note records the teacher-reference alignment fix for context-limited Core ML benchmark runs.

## Problem

`artifacts/benchmarks/minicpm5-teacher-references-full.json` was generated from full prompts. Context-16 Core ML artifacts only see the last 16 prompt tokens after truncation/padding, so longer benchmark prompts were being compared against a different teacher context.

The symptom appeared when the category-balanced int8 run produced low agreement despite using the int8 baseline:

```text
swift run WatchLMBenchmark ... --teacher minicpm5-teacher-references-full.json --prompt-ids zh-short-001,en-short-001,code-fix-001,watch-utility-001,safety-refusal-001 --max-new-tokens 2
result: prompts 5/5, avg_token_agreement 0.5
```

## Change

- Added `--context-tokens` to `tools/benchmark/generate-teacher-references.py`.
- When provided, teacher generation left-truncates encoded prompt tensors to the same context window as the runtime.
- Added `contextTokens` to the generated sidecar metadata.
- Updated the CLI schema smoke test to cover `--context-tokens`.

## Verification

```text
node --test test/teacherReferencesCli.test.js: 1 test passed
.venv/bin/python -m py_compile tools/benchmark/generate-teacher-references.py: passed
generated: artifacts/benchmarks/minicpm5-teacher-references-context16-full.json
contextTokens: 16
promptCount: 10
totalReferenceTokens: 453
```

## Teacher Prefixes

```text
zh-short-001        [18487, 45105]
en-short-001        [416, 4245]
code-fix-001        [3342, 801]
watch-utility-001   [354, 2305]
safety-refusal-001  [1974, 220]
```

## Result

Context-limited Core ML benchmarks now have a matching PyTorch teacher sidecar. Future context-256 and context-512 runs should generate their own sidecars rather than reusing a full-context or context-16 sidecar.
