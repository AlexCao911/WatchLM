# Stateful Step Importance-Guided V-Low4 Int4

Date: 2026-05-30

## Scope

This note records the first Core ML artifact generated from the activation
importance collector rather than a manually chosen layer window.

The candidate follows the safest attention axis found so far: V projection only.
Instead of testing a contiguous mid-layer window, it quantizes the four lowest
activation-energy attention-V layers from the 12-prompt calibration run while
leaving embeddings, norms, lm_head, Q/K/O projections, FFN, and KV state in
fp16.

## Policy

Policy file:

```text
tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-int4.json
```

Policy id:

```text
stateful-step-importance-attention-v-low4-int4-rest-fp16
```

Selected int4 tensors:

```text
model_model_layers_5_self_attn_v_proj_weight
model_model_layers_6_self_attn_v_proj_weight
model_model_layers_7_self_attn_v_proj_weight
model_model_layers_8_self_attn_v_proj_weight
```

Compression audit:

```text
int4 selected ops: 4
selected component: attentionV
selected layers:    5, 6, 7, 8
rejected ops:       1478
```

## Artifacts

Generated:

```text
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-int4/stateful-step-kv-256-mixed.mlpackage
artifacts/coreml/compiled-macos-stateful-step-kv-256-importance-attention-v-low4-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/compiled-watchos11-stateful-step-kv-256-importance-attention-v-low4-int4/stateful-step-kv-256-mixed.mlmodelc
artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-int4/conversion-report.json
```

Size:

```text
mlpackage bytes: 2,159,812,818
artifact dirs:   2.0G mlpackage, 2.0G macOS compiled, 2.0G watchOS 11 compiled
```

The artifact is still a sensitivity probe, not a Watch SE deployable package.
Only four V projection tensors are compressed, so it does not materially solve
the model-size problem.

## Build Gates

Conversion:

```text
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph stateful-step-kv \
  --context-tokens 256 \
  --source-mlpackage artifacts/coreml/real-minicpm5-stateful-step-kv-256/stateful-step-kv-256.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-stateful-step-importance-attention-v-low4-int4.json \
  --output-dir artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-int4
```

Result:

```text
compress_coreml_weights_mixed: succeeded in 76.299s
```

watchOS compile:

```text
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun coremlc compile \
  artifacts/coreml/real-minicpm5-stateful-step-kv-256-importance-attention-v-low4-int4/stateful-step-kv-256-mixed.mlpackage \
  artifacts/coreml/compiled-watchos11-stateful-step-kv-256-importance-attention-v-low4-int4 \
  --platform watchOS \
  --deployment-target 11.0
```

Result:

```text
succeeded
```

Compiling with `--deployment-target 10.0` also emitted an artifact but warned
that this model requires watchOS 11.0 or greater.

## Prefix Diagnostics

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low4-int4-prefix-logits-diagnostics.json
```

Sensitivity report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low4-int4-sensitivity.json
```

Compared with fp16:

```text
gate_ok: true
average prefill top-k overlap: 0.94
prefill top-1 agreement: 1.0

prefix 1:  prefill overlap 5/5, decode overlap 5/5
prefix 2:  prefill overlap 5/5, decode overlap 5/5
prefix 4:  prefill overlap 4/5, decode overlap 5/5
prefix 8:  prefill overlap 5/5, decode overlap 5/5
prefix 12: prefill overlap 5/5, decode overlap 5/5
prefix 16: prefill overlap 4/5, decode overlap 5/5
prefix 18: prefill overlap 5/5, decode overlap 4/5
```

## Teacher Batch Gate

Report:

```text
artifacts/benchmarks/stateful-step-kv-256-importance-attention-v-low4-int4-batch10-cap2.json
```

Result:

```text
prompts:                 10/10 succeeded
average token agreement: 0.9
first token avg:         213.27 ms
decode throughput:       83.60 tok/s
peak RSS:                2181.17 MB
thermal states:          nominal
```

The agreement profile matches the fp16 batch10 cap2 baseline. The only 0.0
prompt is still `watch-utility-002`, which also fails under fp16 because the
runtime stops without emitting the teacher EOS token.

## Interpretation

This candidate is a useful signal:

```text
activation-importance-selected low-energy V projections preserve quality gates
at least as well as manually selected V-only windows.
```

It is not a deployable memory solution:

```text
four V projections are too small a fraction of MiniCPM5-1B to move the package
from multi-GB to Watch SE scale.
```

The next useful experiment is not "try random layers." It is to convert the
activation importance data into a broader mixed policy that can compress many
more tensors while protecting the known sensitive families:

```text
protect: embeddings, lm_head, norms, Q/K/O, layer4 FFN outlier, late layers
test:    more V projections, low-energy FFN subfamilies, grouped-channel int4
gate:    Swift prefix sensitivity + batch10 teacher agreement before promotion
```
