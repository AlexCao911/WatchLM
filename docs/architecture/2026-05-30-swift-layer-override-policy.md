# Swift Layer Override Policy

Date: 2026-05-30

## Scope

This note records the Swift runtime contract update for mixed precision layer
overrides.

The conversion policy files already support `layerOverrides`, and the current
importance-guided V-low4 artifact depends on them:

```text
attentionV layers 5, 6, 7, 8 -> int4
all other components/layers  -> fp16
```

Before this change, Swift `MixedPrecisionPolicy` only understood per-component
global precision. That meant Swift-side manifest/runtime reasoning could not
faithfully represent a real Core ML artifact produced by the conversion path.

## Runtime Behavior

`QuantizationInfo` now decodes and encodes optional `layerOverrides` from normal
JSON object form:

```json
{
  "layerOverrides": {
    "attentionV": {
      "5": "int4",
      "6": "int4"
    }
  }
}
```

`MixedPrecisionPolicy.precision(for:layer:)` resolves precision in this order:

```text
1. layer override for transformer component, if present
2. component default precision
3. edge-layer protection raises int4 to at least int8
```

Layer overrides are limited to:

```text
attentionQKO
attentionV
ffn
```

Unsupported components, out-of-range layers, and invalid precision strings are
rejected before runtime use.

## Validation

The JS manifest validator now mirrors the Swift/runtime contract:

```text
quantization.weights must include attentionV
weights precision must be fp16, int8, or int4
layerOverrides must target transformer components only
layerOverrides layer keys must be integers within architecture.layers
layerOverrides precision must be fp16, int8, or int4
```

## Evidence

Swift targeted test:

```text
swift test --filter MixedPrecisionPolicy
```

Result:

```text
7 tests passed
```

JS targeted validation:

```text
node --test test/modelManifest.test.js
```

Result:

```text
23 tests passed
```

## Consequence

Swift can now consume the same layer-override policy that conversion uses for
the real V-low4 Core ML artifact. This closes a manifest/runtime mismatch:

```text
conversion policy -> Core ML artifact -> Swift policy interpretation
```

The next optimization policy can therefore be generated with layer-level or
component-level overrides without losing fidelity when loaded into the Swift
side of the inference chain.
