# Importance-Guided Policy Candidates

Date: 2026-05-30

## Scope

This note records the first policy candidate generated from the activation
importance report instead of a manual layer sweep.

## Generator

```text
tools/conversion/suggest-importance-policy.py
```

The generator reads an activation importance report and emits an existing
mixed-precision policy JSON. It is intentionally conservative:

```text
all components default to fp16
only one requested component is selected for int4
first and last edge layers are excluded
caller can exclude extra high-risk layers
candidateEvidence records the ranking input
```

## First Candidate

Command:

```bash
.venv/bin/python tools/conversion/suggest-importance-policy.py \
  --importance-report artifacts/benchmarks/minicpm5-activation-importance-cal12.json \
  --component attentionV \
  --candidate-count 4 \
  --protected-edge-layer-count 4 \
  --exclude-layers 4 \
  --policy-id stateful-step-importance-attention-v-low4-int4-rest-fp16 \
  --output tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-int4.json
```

Output policy:

```text
tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-int4.json
```

Selected `attentionV` int4 layers:

```text
layer6: 107708.40625
layer7: 116299.71875
layer5: 119620.703125
layer8: 124729.484375
```

Excluded layers:

```text
0, 1, 2, 3: protected early edge layers
4: FFN activation outlier in cal12
20, 21, 22, 23: protected late edge/high-energy layers
```

The generated policy was validated with:

```bash
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-int4.json \
  --describe-compression-policy
```

The descriptor reports only one int4 compression pass:

```text
precision: int4
opNamePatterns: attention.wv, self_attn.v_proj
layers: 5, 6, 7, 8
```

## Interpretation

This is not yet a deployable artifact. It is the first generated candidate to
feed into the existing Core ML conversion and Swift sensitivity scorer loop.

The reason to try this candidate is clear:

```text
attention V is the locally safest attention family from previous experiments
cal12 ranks layers 5-8 as the lowest non-edge V candidates after excluding the
layer4 FFN outlier
the policy leaves Q/K/O, FFN, norms, embeddings, and lm_head protected
```

## Next Step

Convert this policy into a real stateful-step Core ML artifact, run prefix
diagnostics against the fp16 baseline, and score it with the Swift quantization
sensitivity scorer before considering any wider policy.
