# Split-FFN Importance Candidates

Date: 2026-05-31

## Question

Can the next Watch SE2/SE3 Core ML candidate reduce more memory than the known
safe `attentionV low8` policy without blindly widening int4?

## What Changed

The activation report was regenerated with FFN split into:

```text
ffnGateUp -> mlp.gate_proj + mlp.up_proj
ffnDown   -> mlp.down_proj
```

Report:

```text
artifacts/benchmarks/minicpm5-activation-importance-cal12-split-ffn-groups.json
```

The old report only exposed broad `ffn`, which was too coarse for a useful MLP
quantization decision.

## Key Signal

`ffnDown` is not uniformly safe. It has very concentrated outlier layers:

```text
layer 3:  top-column fraction 0.831
layer 4:  top-column fraction 0.563
layer 22: top-column fraction 0.603
layer 23: top-column fraction 0.335
```

Those layers should stay protected for now. This confirms the direction: FFN
must be selected by subcomponent, layer, and channel concentration, not by a
global FFN rule.

## New Candidate Policies

Generated policies:

```text
tools/conversion/mixed-precision-policy-stateful-step-importance-ffn-gateup-low4-lowrisk-int4.json
tools/conversion/mixed-precision-policy-stateful-step-importance-ffn-down-low4-lowrisk-int4.json
tools/conversion/mixed-precision-policy-stateful-step-importance-v8-ffn-gateup4-down4-lowrisk-int4.json
```

Selection rules:

```text
protected edge layers: 0-3 and 20-23
max top-column fraction: 0.05
max module weighted risk: 0.20
max layer weighted risk: 0.08
```

Selected FFN layers:

```text
ffnGateUp int4: 5, 6, 7, 9
ffnDown   int4: 6, 7, 9, 10
```

The combined candidate reuses the already full-calibration-passing
`attentionV low8` ingredient:

```text
attentionV int4: 6, 7, 8, 9, 10, 11, 12, 13
```

## Risk Gate Results

`ffnGateUp low4`:

```text
gate:                 pass
scored modules:       8
max module risk:      0.015
max layer risk:       0.030
```

`ffnDown low4`:

```text
gate:                 pass
scored modules:       4
max module risk:      0.001
max layer risk:       0.001
```

Combined `attentionV low8 + ffnGateUp low4 + ffnDown low4`:

```text
gate:                 pass
scored modules:       20
max module risk:      0.043
max layer risk:       0.061
```

## Interpretation

This gives us a principled next Core ML candidate. It is not a final deployment
claim yet. The correct next step is:

```text
convert combined policy -> run Swift/CoreML prefix diagnostics -> run full
calibration prompts -> measure artifact size and host/watchOS memory
```

If this candidate holds quality, it becomes the base for expanding FFN coverage.
If it fails, the risk report points at the exact layers/components to roll back
instead of restarting a blind search.
