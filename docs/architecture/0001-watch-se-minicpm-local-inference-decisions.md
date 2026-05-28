# ADR 0001: Apple Watch SE MiniCPM5 Local Inference Decisions

Date: 2026-05-29
Status: Accepted for design review
Related spec: `docs/superpowers/specs/2026-05-29-watch-se-minicpm-inference-design.md`

## Context

The project goal is to run `openbmb/MiniCPM5-1B` locally on Apple Watch SE-class hardware while preserving as much of the original model capability as possible. Apple Watch SE is a difficult target because the device has strict memory, energy, packaging, and runtime constraints. The design therefore needs explicit architectural choices and fallback order, so implementation does not drift toward easier but lower-fidelity solutions.

## Decision 1: Core ML Is the Primary Runtime

Use Core ML `mlprogram` artifacts as the production inference path.

Reasoning:

- Core ML can target Apple silicon accelerators and supports compressed weights and transformer-oriented execution patterns.
- watchOS does not expose the same practical GPU/Metal route available on iPhone, iPad, Mac, Apple TV, and Vision Pro.
- CPU-only GGUF through llama.cpp is useful for baseline experiments and diagnostics, but it is unlikely to meet a usable speed and energy envelope on Apple Watch SE.

Consequences:

- The repository needs host-side conversion tooling.
- Runtime abstractions must be written so Core ML model invocation can be tested independently from SwiftUI.
- Conversion failure is a first-class project risk, not a reason to silently switch to a weaker architecture.

Fallback:

- Use a tiny Core ML smoke-test model to validate app/runtime plumbing.
- Use llama.cpp/GGUF only as a diagnostic comparison, not as the product path.

## Decision 2: Preserve MiniCPM5-1B Architecture First

The first real model artifact must preserve MiniCPM5-1B depth, hidden size, attention structure, tokenizer, and vocabulary.

Reasoning:

- The user explicitly wants to preserve model performance as much as possible.
- Layer pruning, hidden-size reduction, and vocabulary pruning change model behavior more directly than mixed precision and cache/runtime optimization.
- A fidelity-first baseline gives a reference point before any structural fallback is considered.

Consequences:

- The first artifact may be too slow or too large for SE 2.
- Performance work must start with quantization, static-shape decode, KV cache, and lm_head acceleration rather than model shrinkage.
- Any later structural reduction must be justified by benchmark evidence.

Fallback order:

1. Reduce context and response limits.
2. Adjust mixed precision layer policy.
3. Optimize lm_head without hard vocabulary pruning.
4. Add speculative decoding.
5. Consider vocabulary pruning only if measured SE 2 constraints force it.
6. Consider layer pruning only after all above options fail.

## Decision 3: Split Prefill and Decode

Represent inference as two model entry points:

- `prefill`: consumes the prompt and initializes KV cache.
- `decode`: consumes one token and updates KV cache.

Reasoning:

- Autoregressive generation is dominated by repeated one-token decode.
- Separate graphs allow static shapes and easier benchmarking.
- KV cache correctness can be tested independently from prompt ingestion.

Consequences:

- Conversion tooling is more complex than a single forward graph.
- Benchmarking must track prefill latency, first token latency, and steady-state decode separately.

Fallback:

- If stateful KV cache is blocked, use explicit KV input/output tensors and measure copy overhead.

## Decision 4: Use Static Context Variants

Compile separate model variants for fixed context capacities.

Initial variants:

- 256 tokens
- 512 tokens
- 1024 tokens

Reasoning:

- Static shapes are usually easier for Core ML to optimize.
- Apple Watch SE should not target long-context behavior.
- Small variants make it possible to compare memory and speed tradeoffs cleanly.

Default:

- SE 3: 512 tokens
- SE 2: 256 or 512 tokens depending on measured memory and thermal behavior

Fallback:

- Disable 1024-token variant on SE 2 if memory or load time is unacceptable.

## Decision 5: Mixed Precision Beats Uniform Low-Bit Quantization

Use layer-wise mixed precision instead of uniform 3-bit or 4-bit quantization.

Initial policy:

- Embedding: int8 or fp16
- lm_head: int8 or mixed int8/int4
- Norms: fp16
- First and last two transformer layers: int8 where sensitivity requires it
- Attention Q/K/O: int8 until sensitivity tests prove int4 is safe
- FFN up/gate/down: int4 first
- KV cache: int8 first

Reasoning:

- 1B-class models are more vulnerable to aggressive quantization than larger models.
- Keeping sensitive components at higher precision protects instruction following and next-token distribution.
- Mixed precision creates a measurable tuning surface.

Consequences:

- The conversion pipeline must support per-layer or per-op quantization configuration.
- The benchmark suite must include quality drift metrics, not only latency.

Fallback:

- If the artifact exceeds memory limits, reduce precision only after sensitivity reports identify low-impact layers.

## Decision 6: Treat lm_head as the First Structural Bottleneck

If full-architecture mixed precision is too slow, optimize `lm_head` before changing transformer layers.

Preferred order:

1. Full vocabulary int8 lm_head.
2. Mixed int8/int4 lm_head.
3. Low-rank lm_head factorization with distillation recovery.
4. Candidate-token scoring with full-score fallback.
5. Hard vocabulary pruning only as a last resort.

Reasoning:

- MiniCPM5-1B uses a large vocabulary and untied embeddings.
- lm_head can dominate one-token decode on small hardware.
- lm_head changes are easier to recover through distillation than pruning transformer layers.

Consequences:

- The model quality harness must compare top-k token agreement and distribution drift.
- Low-rank or candidate-token modes need a clear fallback path when uncertainty is high.

## Decision 7: Benchmark Evidence Controls Fallbacks

No fallback may replace the fidelity-first path without device or conversion evidence.

Required evidence:

- Artifact size
- Model load time
- Prefill latency
- First token latency
- Decode tokens per second
- Peak resident memory
- Thermal state over five short turns
- Quality drift against BF16 teacher

Reasoning:

- Apple Watch performance is highly dependent on runtime scheduling and thermal limits.
- It is easy to prematurely choose a smaller model that passes superficial tests but fails the user's fidelity goal.

Consequences:

- The implementation must create a benchmark harness before serious model optimization.
- Reports should be saved as artifacts so decisions remain auditable.

## Decision 8: Asset Installation Stays Outside the Main Bundle

Large model artifacts are installed after app installation rather than bundled in the watchOS app.

Reasoning:

- The watchOS app bundle size limit is far below the expected model artifact size.
- Separate assets allow SE 2 and SE 3 variants without bloating the executable.

Consequences:

- The app needs a model asset manager before real inference can be productized.
- Manifests, hashes, and recovery paths are required.

Fallback:

- Support developer sideloading for early experiments.
- Add production asset delivery only after artifact size and App Store constraints are understood.

## Risk Matrix

| Risk | Likelihood | Impact | Detection | Mitigation |
| --- | --- | --- | --- | --- |
| Core ML conversion cannot express MiniCPM5 decode with KV cache | Medium | High | Conversion test fails or logits mismatch | Use explicit KV tensors; simplify graph boundaries; validate smaller smoke-test model first |
| watchOS Core ML cannot execute the converted artifact efficiently | High | High | Device benchmark below token/sec target | Mixed precision tuning; context reduction; lm_head optimization; speculative decoding |
| SE 2 memory budget cannot fit full architecture | High | High | Load failure, jetsam, or peak memory above safe threshold | 256 context variant; stricter mixed precision; lm_head optimization; last-resort vocabulary pruning |
| Quantization damages model quality | Medium | High | Top-k agreement and KL drift exceed threshold | Preserve sensitive layers; use calibration set; run recovery distillation |
| Large model asset cannot be distributed cleanly | Medium | Medium | Install or App Store review constraints block model delivery | Developer sideload first; asset manifests; production review plan later |
| Sustained inference causes thermal throttling | High | Medium | Five-turn benchmark slows or invalidates session | Short output limits; foreground-only generation; thermal degraded mode |
| Tokenizer implementation diverges from MiniCPM chat template | Medium | High | Prompt/token round-trip tests fail | Preserve upstream tokenizer assets; snapshot chat-template tests |
| Speculative decoding adds complexity without speedup | Medium | Medium | Draft/verifier benchmark shows low acceptance or memory pressure | Make speculative decoding phase-two only; keep baseline decode path |

## Quality Gates

Before moving from one phase to the next:

1. Documentation phase: design and ADR documents are committed and self-reviewed.
2. Planning phase: implementation plan maps every design decision to concrete tasks.
3. Tooling phase: host tests validate manifests, prompt fixtures, and metadata extraction.
4. App shell phase: watch app runs with mock streaming and missing-model UI.
5. Runtime phase: Core ML smoke-test artifact loads and produces timed output.
6. MiniCPM phase: first MiniCPM artifact loads on target hardware or records a reproducible blocker.
7. Optimization phase: each fallback is justified by benchmark evidence.

## Rejected Alternatives

### Use iPhone as the Main Inference Device

Rejected because the user wants Apple Watch-only local inference.

### Start With a Distilled 100M-300M Model

Rejected as the first path because it weakens the model-performance preservation goal.

### Use Uniform Q3/Q4 Everywhere

Rejected as the default because uniform low-bit compression is likely to harm a 1B model's behavior more than mixed precision.

### Prune Vocabulary Immediately

Rejected as the default because it changes model coverage and multilingual behavior. It remains a last-resort SE 2 fallback.

### Build Product UI Before Benchmark Infrastructure

Rejected because architecture choices depend on device evidence, and a polished UI would hide unresolved runtime risk.
