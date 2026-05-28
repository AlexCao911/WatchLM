# WatchLM

WatchLM is a staged implementation project for running `openbmb/MiniCPM5-1B` locally on Apple Watch SE-class hardware.

The first executable layer is host-side validation for the artifacts that later Core ML conversion and watchOS runtime work will consume:

- model manifests that preserve the MiniCPM5-1B architecture contract.
- benchmark prompts that cover Chinese, English, coding, watch utility, and safety probes.
- benchmark reports with SE 2 and SE 3 usability gates.

## Local Validation

Run all current tests:

```sh
node --test
```

Validate individual evidence files:

```sh
node bin/watchlm-validate.js manifest fixtures/sample-model-manifest.json
node bin/watchlm-validate.js prompts fixtures/benchmark-prompts.json
node bin/watchlm-validate.js report fixtures/sample-benchmark-report.json
```

Validate all host-side contracts:

```sh
node bin/watchlm-validate.js all \
  --manifest fixtures/sample-model-manifest.json \
  --prompts fixtures/benchmark-prompts.json \
  --report fixtures/sample-benchmark-report.json
```

## Artifact Policy

Real `.mlpackage`, compiled `.mlmodelc`, GGUF, and checkpoint files are intentionally not committed. They will be generated or installed outside the main watchOS app bundle and represented in git by manifests, conversion contracts, and benchmark reports.
