# Core ML Conversion Contract

This directory stores the auditable contract for generated Core ML model artifacts. It does not store model files.

The first real MiniCPM5 artifact must preserve:

- source checkpoint id `openbmb/MiniCPM5-1B`.
- original tokenizer and vocabulary.
- split `prefill` and `decode` model entry points.
- fixed context variants from `256`, `512`, and `1024`.
- the fidelity-first mixed precision policy before structural fallback.

Generated artifacts belong under `artifacts/`, which is ignored by git:

- `.mlpackage`
- `.mlmodelc`
- `.gguf`
- `.safetensors`
- generated benchmark reports

When conversion produces a real artifact, update `coreml-artifact-contract.json` with the checkpoint revision or checksum, tokenizer checksum, artifact paths, quantization policy id, and logits validation summary.

## Real MiniCPM5 Prefill Spike

The current real-model spike converts a fixed `prefill-16` graph first. It intentionally externalizes `position_ids` and a 4D additive causal mask so Core ML does not need to convert Transformers' dynamic mask helper ops.

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-16 \
  --compute-precision float16 \
  --compression none
```

Generate the int8 Core ML package:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-16-int8 \
  --compute-precision float16 \
  --compression int8
```

Generate the full-model int4 Core ML package:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-16-int4 \
  --compute-precision float16 \
  --compression int4
```

The first full-model int4 package compiled for watchOS and reduced the package to about 516MB, but the single-prompt logits check did not preserve top-1. Treat it as a compression proof, not as the production fidelity policy.

Inspect the fidelity-first mixed precision plan without loading the model:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy.json \
  --describe-compression-policy
```

Generate mixed-compressed graphs with the same policy:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill-kv \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-kv-16-mixed \
  --compute-precision float16 \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy.json

.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph decode \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-decode-16-mixed \
  --compute-precision float16 \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy.json
```

The mixed policy keeps embeddings, lm head, attention, and protected edge-layer FFNs at int8 or fp16, while allowing middle FFN projections to use int4 palettization. This is the current default before considering structural reduction or int2.

For a conservative first pass, use an explicit layer override policy that only palettizes layer 12 FFN:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-ffn12.json \
  --describe-compression-policy
```

If the fp16 graph already exists, compress it directly without re-running PyTorch tracing and Core ML conversion:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --source-mlpackage artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage \
  --compression int8 \
  --output-dir artifacts/coreml/real-minicpm5-decode-16-int8

.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --source-mlpackage artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy.json \
  --output-dir artifacts/coreml/real-minicpm5-decode-16-mixed
```

Current context-16 evidence:

- `prefill-kv-16-int8` and `decode-16-int8` compile for watchOS and preserve the teacher top-1.
- `prefill-kv-16-mixed` and `decode-16-mixed` shrink to about 830MB each, but the current middle-FFN int4 policy does not preserve top-1.
- `prefill-kv-16-mixed-ffn12` and `decode-16-mixed-ffn12` compile for watchOS and preserve teacher top-1; they only shrink each graph to about 1.072GB, so the next search should widen the int4 layer override window gradually.
- `prefill-kv-16-mixed-ffn10-13` and `decode-16-mixed-ffn10-13` compile for watchOS and preserve teacher top-1 on the local prompt. They shrink each graph to about 1.040GB, with higher drift than FFN12: prefill top10 agreement 9/10, decode top10 agreement 9/10, decode KV max error 0.96533203125.
- `prefill-kv-16-mixed-ffn8-15` and `decode-16-mixed-ffn8-15` compile for watchOS and preserve teacher top-1 on the local prompt. They shrink each graph to about 998MB, but quality drift rises again: prefill top10 agreement 9/10, decode top10 agreement 8/10, decode KV max error 1.501708984375. Treat this as a boundary point, not the default.

Generate the FFN10...13 policy variant from existing fp16 packages:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill-kv \
  --source-mlpackage artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-ffn10-13.json \
  --output-dir artifacts/coreml/real-minicpm5-prefill-kv-16-mixed-ffn10-13

.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph decode \
  --source-mlpackage artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-ffn10-13.json \
  --output-dir artifacts/coreml/real-minicpm5-decode-16-mixed-ffn10-13
```

Generate the more aggressive FFN8...15 boundary variant from existing fp16 packages:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill-kv \
  --source-mlpackage artifacts/coreml/real-minicpm5-prefill-kv-16/prefill-kv-16.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-ffn8-15.json \
  --output-dir artifacts/coreml/real-minicpm5-prefill-kv-16-mixed-ffn8-15

.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph decode \
  --source-mlpackage artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage \
  --compression mixed \
  --precision-policy tools/conversion/mixed-precision-policy-ffn8-15.json \
  --output-dir artifacts/coreml/real-minicpm5-decode-16-mixed-ffn8-15
```

Generate prefill and decode graphs with explicit KV cache IO:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph prefill-kv \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-kv-16 \
  --compute-precision float16 \
  --compression none

.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --graph decode \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-decode-16 \
  --compute-precision float16 \
  --compression none
```

Compile the quantized package for watchOS:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun coremlc compile \
  artifacts/coreml/real-minicpm5-prefill-16-int8/prefill-16-int8.mlpackage \
  artifacts/coreml/compiled-watchos-int8 \
  --platform watchOS \
  --deployment-target 10.0
```

Validate logits against the PyTorch teacher:

```sh
TMPDIR="$PWD/artifacts/tmp" \
  .venv/bin/python tools/validation/validate-coreml-prefill.py \
  --mlpackage artifacts/coreml/real-minicpm5-prefill-16-int8/prefill-16-int8.mlpackage \
  --context-tokens 16 \
  --report artifacts/coreml/real-minicpm5-prefill-16-int8/logits-validation.json
```

Validate decode logits and one-token KV outputs against the PyTorch teacher:

```sh
TMPDIR="$PWD/artifacts/tmp" \
  .venv/bin/python tools/validation/validate-coreml-decode.py \
  --mlpackage artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage \
  --context-tokens 16 \
  --report artifacts/coreml/real-minicpm5-decode-16/decode-validation.json
```
