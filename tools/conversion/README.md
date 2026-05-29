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
