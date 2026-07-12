# SE2 Stateful-Step Candidate Manifest

Date: 2026-05-30

## Scope

This note records the first manifest-level Watch SE2 context-256 candidate for
the shared `stateful-step-kv` Core ML route.

It is separate from quantization diagnosis notes. This document is about which
artifact shape the Swift runtime and validation tooling should select for SE2.

## Candidate Fixture

Manifest fixture:

```text
tools/validation/fixtures/watch-se2-stateful-step-model-manifest.json
```

SE2 selected artifact:

```text
context:       256
graph:         stateful-step-kv
prefill path:  Models/MiniCPM5/stateful-step-kv-256-int4.mlpackage
decode path:   Models/MiniCPM5/stateful-step-kv-256-int4.mlpackage
KV route:      stateful-preferred
KV precision:  fp16
```

The prefill and decode paths intentionally match. This keeps the manifest
aligned with the Swift runtime contract: one shared `MLModel` instance owns one
`MLState`.

## Conversion Contract

The top-level conversion contract now points at the same stateful-step shape:

```text
tools/conversion/coreml-artifact-contract.json
```

The SE2 artifact evidence remains outside git under `artifacts/`, but the
contract points to the generated stateful-step package and compiled watchOS
artifact paths.

## Current Evidence

Existing host benchmark reports:

```text
artifacts/benchmarks/stateful-step-kv-256-int4-teacher-smoke.json
artifacts/benchmarks/stateful-step-kv-256-int4-smoke.json
```

Observed:

```text
shared artifact total:        551,309,343 bytes
teacher-smoke peak RSS:       651.25 MB
non-teacher smoke peak RSS:   953.09 MB
teacher token agreement:      0.0
```

This is a materially better memory shape than the split int4 route, which
peaked around 3 GB to 3.6 GB on host.

## Status

This is a deployability candidate, not a quality-promoted model.

The current all-int4 stateful-step candidate proves the Swift/Core ML chain can
select and run a shared graph shape that is much closer to SE2 memory needs. It
does not yet preserve MiniCPM5 behavior well enough: quality is blocked by token
agreement.

## Next Work

Continue quantization search on the shared stateful-step graph:

```text
1. keep final layers and lm_head out of int4
2. isolate earlier layers or smaller component groups
3. compare prefix logits against fp16 before running longer smoke prompts
4. promote only policies that keep agreement while staying under the SE2 memory gate
```
