# Activation-Weighted Quantization Risk Gate

Date: 2026-05-31

## Purpose

The community evidence refresh pointed to AWQ/imatrix-style calibration
signals: low-bit policies should be filtered with activation evidence before
spending time on Core ML conversion.

This note records the first executable version of that idea:

```text
tools/conversion/score-quantization-risk.py
```

It is an offline conversion/validation tool, not watch runtime code. The Swift
runtime still consumes the selected Core ML artifact, manifest, tokenizer, KV
state, logits, and benchmark reports.

## Inputs

The scorer consumes:

```text
activation importance report:
  artifacts/benchmarks/minicpm5-activation-importance-cal12-groups.json

mixed precision policy:
  tools/conversion/mixed-precision-policy-*.json
```

It evaluates only modules whose effective policy precision is lower than fp16.
Current risk multipliers:

```text
int8: 0.35
int4: 1.00
int3: 1.40
int2: 2.00
```

## Scoring

Per low-bit module:

```text
componentActivationEnergyFraction =
  module total activation energy / total activation energy for that component

topColumnEnergyFraction =
  most active input channel / module total activation energy

topGroupEnergyFraction =
  most active channel group / module total activation energy

weightedRiskScore =
  precisionRiskMultiplier
  * componentActivationEnergyFraction
  * (1 + topColumnEnergyFraction + topGroupEnergyFraction)
```

The scorer also aggregates low-bit module scores per transformer layer. This is
important because local evidence showed that individually safe axes can fail
when stacked in the same layers. The known failed V + QK/O policy is caught by
this layer aggregate gate.

## Commands

Safe V-low8 policy:

```bash
.venv/bin/python tools/conversion/score-quantization-risk.py \
  --importance-report artifacts/benchmarks/minicpm5-activation-importance-cal12-groups.json \
  --precision-policy tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low8-lowrisk-int4.json \
  --max-weighted-risk 0.2 \
  --max-layer-weighted-risk 0.08 \
  --max-top-column-fraction 0.05 \
  --output artifacts/benchmarks/stateful-step-importance-attention-v-low8-lowrisk-risk-score.json
```

Known failed V + QK/O policy:

```bash
.venv/bin/python tools/conversion/score-quantization-risk.py \
  --importance-report artifacts/benchmarks/minicpm5-activation-importance-cal12-groups.json \
  --precision-policy tools/conversion/mixed-precision-policy-stateful-step-layer8-15-v-layer11-12-qk-int4.json \
  --max-weighted-risk 0.2 \
  --max-layer-weighted-risk 0.08 \
  --max-top-column-fraction 0.05 \
  --output artifacts/benchmarks/stateful-step-layer8-15-v-layer11-12-qk-risk-score.json
```

## Results

V-low8:

```text
gate:                 pass
scored modules:       8
rejected modules:     0
rejected layers:      0
max module risk:      0.043
max layer risk:       0.043
```

V + QK/O:

```text
gate:                 fail
scored modules:       14
rejected modules:     0
rejected layers:      1
max module risk:      0.057
failing layer:        layer 12
layer 12 risk:        0.085 > 0.080
```

## Interpretation

Module-level risk alone would not catch the failed V + QK/O policy. Layer-level
accumulation does catch it, matching the earlier Core ML prefix-collapse
evidence.

This makes the scorer useful as a pre-export rejection gate:

```text
safe ingredient:
  V-low8 remains low risk and passed the full calibration-prefix Swift gate.

unsafe composition:
  V + QK/O exceeds same-layer aggregate risk before Core ML conversion.
```

## Limitations

This is not yet full activation-weighted reconstruction error. It does not
quantize weights and compare reconstructed outputs. It is a calibration-derived
risk heuristic that combines activation energy, channel concentration, group
concentration, precision depth, and same-layer accumulation.

The next improvement should add optional weight reconstruction error per tensor
or channel group, then multiply that error by the same activation statistics.

## Next Use

Before generating the next OptiQ-shaped mixed 4/8/fp16 Core ML candidate:

1. Generate the policy candidate.
2. Run this risk gate with the fixed calibration report.
3. Reject candidates that fail module, channel, group, or layer aggregate
   thresholds.
4. Only then spend time on Core ML conversion, watchOS compile, Swift prefix
   diagnostics, and benchmark gates.
