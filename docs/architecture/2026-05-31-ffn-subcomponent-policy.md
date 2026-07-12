# FFN Subcomponent Precision Policy

Date: 2026-05-31

## Purpose

The next quantization direction is an OptiQ-shaped mixed 4/8/fp16 policy, not a
global int4 sweep. That requires finer control than the previous `ffn` component
allowed.

Before this change, a policy could only select:

```text
ffn -> mlp.gate_proj + mlp.up_proj + mlp.down_proj
```

That is too coarse for MiniCPM5-1B because local evidence already showed
uncalibrated FFN-wide int4 can fail. Community recipes often treat MLP
subfamilies asymmetrically, so WatchLM needs to control them separately.

## New Control Surface

Mixed precision policies may now optionally declare:

```json
{
  "weights": {
    "ffn": "fp16",
    "ffnGateUp": "int4",
    "ffnDown": "int8"
  },
  "layerOverrides": {
    "ffnGateUp": {
      "12": "fp16"
    },
    "ffnDown": {
      "12": "int4"
    }
  }
}
```

Semantics:

```text
ffnGateUp -> mlp.gate_proj + mlp.up_proj
ffnDown   -> mlp.down_proj
ffn       -> legacy broad FFN fallback
```

If `ffnGateUp` or `ffnDown` is absent, it inherits `ffn`. Existing policies keep
working.

## Implemented Surfaces

Swift runtime manifest policy:

```text
Sources/ModelRuntime/Quant/MixedPrecisionPolicy.swift
Sources/ModelRuntime/Model/ModelManifest.swift
```

Conversion tooling:

```text
tools/conversion/convert-minicpm5-coreml.py
```

Activation collection:

```text
tools/conversion/collect-activation-importance.py
```

Risk scoring:

```text
tools/conversion/score-quantization-risk.py
```

Manifest validation:

```text
tools/validation/modelManifest.js
```

## Compatibility

Legacy policies remain valid:

```json
{
  "weights": {
    "ffn": "int4"
  },
  "layerOverrides": {
    "ffn": {
      "12": "int8"
    }
  }
}
```

When no explicit FFN subcomponent is present, conversion audit names continue to
record legacy `ffn` selections. When a policy explicitly uses `ffnGateUp` or
`ffnDown`, selector audit records those specific components.

## Verification

Targeted tests:

```text
swift test --filter MixedPrecisionPolicy
node --test test/activationImportanceCli.test.js test/realConversionCli.test.js test/modelManifest.test.js test/quantizationRiskScorer.test.js
```

Results:

```text
Swift MixedPrecisionPolicy: 8/8 passed
Node targeted tooling tests: 66/66 passed
```

## Why It Matters

This is a prerequisite for a principled MiniCPM5-1B policy search:

```text
V attention:
  already has a low-risk int4/low8 ingredient

FFN:
  needs subcomponent-specific experiments instead of whole-layer int4

Next candidate:
  ffnGateUp / ffnDown can now be ranked and gated separately with activation
  statistics before Core ML conversion
```

This keeps the next optimization step aligned with the community evidence:
mixed precision, component asymmetry, and calibration-driven protection rather
than blind FFN layer widening.
