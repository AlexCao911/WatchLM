# Watch SE MiniCPM5 Requirements Traceability Matrix

Date: 2026-05-29
Status: Design-stage traceability

## Purpose

This document maps the user goal and architecture decisions to concrete deliverables and evidence. It exists so implementation can stay aligned with the real objective: build toward Apple Watch SE-only local MiniCPM5 inference with rigorous Superpowers documentation, staged development, and commits at each completed part.

## Requirement Sources

- User goal: use Superpowers, do rigorous design documentation, then develop, committing each completed part.
- User constraint: use only Apple Watch to run inference.
- User constraint: hardware target is Apple Watch SE.
- User constraint: optimize quantization and inference structure.
- User constraint: preserve model capability as much as possible.
- Design spec: `docs/superpowers/specs/2026-05-29-watch-se-minicpm-inference-design.md`
- ADR: `docs/architecture/0001-watch-se-minicpm-local-inference-decisions.md`

## Requirement Matrix

| ID | Requirement | Design commitment | Implementation evidence required | Verification method |
| --- | --- | --- | --- | --- |
| R1 | Use Superpowers workflow | Design gate documented in the spec; implementation waits for approval before planning | Design docs, ADR, implementation plan, commits for each completed part | Inspect docs, git log, and plan status |
| R2 | Run locally on Apple Watch only | Non-goal excludes iPhone, cloud, and LAN inference for main path | watchOS app/runtime does not require iPhone or network inference endpoint | Inspect app architecture, runtime configuration, and manual/device test |
| R3 | Target Apple Watch SE hardware | SE 2 and SE 3 profiles are separated | Device profiles in config/manifests; benchmark reports identify SE generation | Inspect manifests and benchmark output |
| R4 | Preserve MiniCPM5-1B capability as much as possible | Original architecture and tokenizer are preserved first; structural reductions are fallback-only | First real artifact keeps MiniCPM5 depth, hidden size, attention structure, tokenizer, and vocabulary | Inspect model manifest and conversion report |
| R5 | Prefer Core ML production runtime | ADR selects Core ML `mlprogram` as primary runtime | Runtime adapter loads Core ML artifact; GGUF is absent or diagnostic-only | Inspect runtime code and tests |
| R6 | Optimize inference structure | Prefill/decode split, static context variants, KV cache | Runtime exposes prefill and one-token decode paths with timing metrics | Unit/integration tests and benchmark report |
| R7 | Optimize quantization without unnecessary quality loss | Mixed precision policy and sensitivity search required | Quantization config supports per-layer policy; report captures quality drift | Inspect config, conversion tests, and drift report |
| R8 | Keep model outside watchOS app bundle | Asset manager owns large model availability | Manifest/hash-based model asset manager exists; app can represent missing-model state | Unit tests and app shell behavior |
| R9 | Provide benchmark evidence before fallbacks | ADR requires evidence for fallback decisions | Benchmark harness records load time, prefill, first token, decode speed, memory, thermal notes, quality drift | Run benchmark fixture/tests; inspect report artifact |
| R10 | Commit completed parts | User explicitly requested commit after each part | Git commits correspond to design, plan, tooling, app shell, runtime, and optimization stages | Inspect git log |
| R11 | Support graceful errors | Design specifies installation and inference error handling | Asset and runtime errors have typed representations and user-visible states | Unit tests and UI state tests |
| R12 | Avoid long-context scope creep | Default context is 512; variants are 256/512/1024 | Context selection clamps to supported variants | Unit tests and benchmark configs |
| R13 | Support cancellation and short foreground sessions | App layer owns cancellation and lifecycle | Runtime can cancel at token boundary; UI exposes cancellation | Unit/integration tests |
| R14 | Preserve auditability | Reports and manifests are first-class artifacts | Generated reports are saved with metadata and source model identifiers | Inspect output schema and sample reports |

## Phase-to-Requirement Coverage

| Phase | Covered requirements | Exit evidence |
| --- | --- | --- |
| Documentation | R1, R3, R4, R5, R6, R7, R8, R9, R10, R12, R14 | Design spec, ADR, traceability matrix, commits |
| Implementation planning | R1, R10, all requirements mapped to tasks | Superpowers implementation plan committed |
| Host tooling | R3, R4, R7, R8, R9, R14 | Metadata inspector, manifests, prompt fixtures, tests |
| watchOS shell | R2, R3, R8, R11, R13 | App launches, missing-model UI, asset-state tests |
| Tokenizer/runtime interfaces | R4, R6, R11, R12, R13 | Tokenizer tests, mock streaming tests, runtime protocol tests |
| Core ML smoke test | R5, R6, R9, R11, R14 | Timed Core ML smoke artifact report |
| MiniCPM artifact integration | R2, R3, R4, R5, R6, R7, R8, R9, R12 | Conversion report, model manifest, device benchmark |
| Optimization | R4, R6, R7, R9, R14 | Mixed precision report, lm_head report, speculative decoding report if used |

## Completion Evidence Checklist

The goal cannot be marked complete until current-state evidence proves all applicable items:

- Design spec exists and is committed.
- ADR exists and is committed.
- Implementation plan exists and is committed after design approval.
- Repository contains implementation artifacts, not only documentation.
- Each completed implementation part has a corresponding git commit.
- Host tooling has tests that pass locally.
- watchOS app shell exists and builds or has a documented reproducible blocker.
- Runtime interfaces are testable without a real model artifact.
- Model manifest and asset-manager behavior are implemented.
- Benchmark schema and at least one sample report exist.
- If a real MiniCPM Core ML artifact is not committed because of size, the repo includes reproducible conversion/packaging commands and a manifest contract.
- Fallback decisions are supported by benchmark evidence rather than convenience.

## Approval Gate

The Superpowers brainstorming approval gate was satisfied by the user message on 2026-05-29: "开始执行 继续实现". Planning and implementation may proceed under the requirement that each completed part is tested and committed separately.
