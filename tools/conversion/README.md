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
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-16 \
  --compute-precision float16
```

Generate the int8 Core ML package:

```sh
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py \
  --context-tokens 16 \
  --output-dir artifacts/coreml/real-minicpm5-prefill-16-int8 \
  --compute-precision float16 \
  --quantize
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
