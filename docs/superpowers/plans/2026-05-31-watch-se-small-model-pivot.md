# Watch SE Small Model Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit sizing gate for smaller distilled model candidates so WatchLM can stop spending conversion time on MiniCPM5-1B policies that cannot plausibly fit Apple Watch SE budgets.

**Architecture:** Keep the Swift/CoreML inference chain as the production runtime, but add host-side planning gates before conversion. Candidate profiles estimate artifact bytes, KV bytes, and peak resident memory for SE2/SE3 context variants; only candidates that pass the planning gate move to Core ML conversion and Swift benchmark.

**Tech Stack:** Node.js validation tooling, JSON fixtures, existing `watchlm-validate` CLI, docs under `docs/architecture` and `docs/benchmarks`.

---

### Task 1: Candidate Sizing Gate

**Files:**
- Create: `tools/validation/modelCandidateSizing.js`
- Create: `tools/validation/fixtures/model-candidates.json`
- Create: `test/modelCandidateSizing.test.js`
- Modify: `tools/validation/watchlm-validate.js`
- Modify: `test/validationCli.test.js`

- [ ] **Step 1: Write failing tests**

Create tests that prove:

```text
1. MiniCPM5-1B fp16/mixed remains over the SE2 planning budget.
2. A 350M distilled int4 candidate can pass SE2 context-256 sizing.
3. Invalid candidate files report all schema errors.
4. watchlm-validate candidates <path> prints a pass/fail summary.
```

Run:

```bash
node --test test/modelCandidateSizing.test.js test/validationCli.test.js
```

Expected before implementation:

```text
ERR_MODULE_NOT_FOUND or unknown candidates command
```

- [ ] **Step 2: Implement the gate**

Create `modelCandidateSizing.js` with:

```text
validateModelCandidateSuite(suite)
evaluateModelCandidate(candidate, deviceProfile)
summarizeCandidateEvaluation(evaluation)
```

The gate estimates:

```text
weight bytes = parameterCount * weightBits / 8 * compressionOverhead
KV bytes = layers * 2 * kvHeads * contextTokens * headDimension * kvBytes
artifact bytes = weight bytes + tokenizer bytes
peak RSS = artifact bytes * coreMLLoadMultiplier + KV bytes + temporaryArena bytes
```

- [ ] **Step 3: Add CLI integration**

Add:

```bash
node tools/validation/watchlm-validate.js candidates tools/validation/fixtures/model-candidates.json
```

Expected:

```text
candidates ok: 3 candidates, 1 passing SE2 gate
```

- [ ] **Step 4: Verify and commit**

Run:

```bash
node --test test/modelCandidateSizing.test.js test/validationCli.test.js
git diff --check
```

Commit:

```bash
git add tools/validation/modelCandidateSizing.js tools/validation/watchlm-validate.js tools/validation/fixtures/model-candidates.json test/modelCandidateSizing.test.js test/validationCli.test.js docs/superpowers/plans/2026-05-31-watch-se-small-model-pivot.md
git commit -m "Add Watch SE model candidate sizing gate"
```

### Task 2: Pivot Decision Record

**Files:**
- Create: `docs/architecture/2026-05-31-small-model-distillation-pivot.md`

- [ ] **Step 1: Record current evidence**

Document:

```text
V-low8: quality passes but artifact remains about 2GB.
global int4: memory improves but quality collapses.
V8 + split-FFN: converts and runs, but full12 sensitivity fails.
```

- [ ] **Step 2: Define the new primary path**

Record:

```text
MiniCPM5-1B becomes teacher/baseline.
Runtime candidates should target 125M-350M first, 600M only if size gates pass.
Distillation and structural reduction are now primary levers.
```

- [ ] **Step 3: Verify and commit**

Run:

```bash
git diff --check
```

Commit:

```bash
git add docs/architecture/2026-05-31-small-model-distillation-pivot.md
git commit -m "Record small model distillation pivot"
```
