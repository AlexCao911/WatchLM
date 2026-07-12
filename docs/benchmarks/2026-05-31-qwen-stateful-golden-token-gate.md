# Qwen Stateful Golden Token Gate

Date: 2026-05-31

## Goal

Add a simple multi-token agreement gate for the Qwen stateful-step path. The
existing top-k diagnostics catches prefill and first decode drift, but it missed
the narrow FFN-down int4 candidate because that candidate matched the first
three generated tokens and drifted on token four.

## Reference Fixture

```text
tools/validation/fixtures/qwen3-watch-utility-002-fp32-compute-int8-reference.json
```

Reference source:

```text
qwen3-stateful-step-fp32-compute-int8-golden
```

Reference tokens for `watch-utility-002`:

```text
[785, 1614, 9329, 374]
```

This is a temporary Core ML golden generated from the current validated
fp32-compute int8 stateful-step artifact. It is not a PyTorch teacher yet, but
it gives the benchmark runner a stable multi-token gate while Qwen conversion
work continues.

## Baseline Check

Artifact:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8/stateful-step-kv-256-int8.mlpackage
```

Report:

```text
artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-fp32-compute-int8-golden-gate.json
```

Result:

```text
generatedTokenIDs: [785, 1614, 9329, 374]
tokenAgreement: 1.0
exactTokenMatchCount: 4
```

## Narrow Int4 Check

Artifact:

```text
artifacts/coreml/qwen3-0.6b-stateful-step-kv-256-fp32-compute-mixed-ffn-down-low4-int4/stateful-step-kv-256-mixed.mlpackage
```

Report:

```text
artifacts/benchmarks/qwen3-0.6b-stateful-step-kv-256-fp32-compute-mixed-ffn-down-low4-int4-golden-gate.json
```

Result:

```text
generatedTokenIDs: [785, 1614, 9329, 702]
tokenAgreement: 0.75
exactTokenMatchCount: 3
firstMismatchIndex: 3
```

## Interpretation

This gives the Qwen branch a practical promotion check:

```text
prefill/first-decode top-k gate
plus
multi-token generated-token agreement
```

The next Qwen compression candidates should be rejected unless they preserve
the full reference token sequence on this prompt, then the fixture should be
expanded to more prompts or replaced by PyTorch teacher references.
