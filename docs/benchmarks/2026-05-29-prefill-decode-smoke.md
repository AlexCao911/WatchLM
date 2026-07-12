# Split Prefill/Decode Smoke Validation

Date: 2026-05-29
Host Xcode: Xcode 26.3, build 17C529
Simulator runtime: watchOS 26.2
Destination: Apple Watch SE 3 (44mm)
Branch: `codex/watch-se-minicpm-foundation`

## Scope

This run validates the next inference layer after the single-model Core ML smoke test:

- split `prefill` and `decode` Core ML models.
- explicit KV-cache handoff between the two models.
- tokenizer abstraction in front of the runtime.
- MiniCPM5 chat-template constants, special token ids, and int8 KV-cache sizing contracts.
- separate SE 2 and SE 3 quantized artifact variants in the manifest contract.

This is still not MiniCPM5-1B inference. The prefill/decode models are tiny deterministic smoke models used to prove the watchOS Core ML execution shape before replacing them with quantized MiniCPM-derived artifacts.

## Commands

```sh
node --test
swift test
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme WatchLM -destination 'platform=watchOS Simulator,name=Apple Watch SE 3 (44mm)'
```

## Results

| Check | Result |
| --- | --- |
| Node host tests | 40 passed, 0 failed |
| macOS Swift tests | 20 passed, 0 failed |
| watchOS SE 3 simulator Swift tests | 20 passed, 0 failed |
| watchOS simulator result bundle | `~/Library/Developer/Xcode/DerivedData/WatchLM-cidoceiepgeomqgqcsornmuwdlpw/Logs/Test/Test-WatchLM-2026.05.29_09-19-47-+0800.xcresult` |

macOS SwiftPM split runtime print:

```text
WATCHLM_PREFILL_DECODE_SMOKE output=DEF load_ms=13.074 prefill_ms=0.997 decode_tps=21739.13
```

Watch SE 3 simulator split runtime print:

```text
WATCHLM_PREFILL_DECODE_SMOKE output=DEF load_ms=28.856 prefill_ms=0.915 decode_tps=6153.85
```

Watch SE 3 simulator single-model Core ML smoke print:

```text
WATCHLM_COREML_SMOKE output=7.5 load_ms=24.946 prediction_ms=1.299
```

## SE 2 / SE 3 Coverage

- SE 2 is covered at the contract layer with a 256-token quantized artifact variant and `maxNewTokens=64`.
- SE 3 is covered at the contract layer with a 512-token quantized artifact variant and `maxNewTokens=96`.
- This Xcode install exposes Apple Watch SE 3 simulators only. There is no Apple Watch SE 2 simulator destination available locally, so SE 2 has not been simulator-executed in this run.
- Physical-device measurements are still required before claiming real SE 2 or SE 3 token speed.

## Current Gap To Real MiniCPM Inference

Implemented:

- Core ML loading and prediction on watchOS simulator.
- Split prefill/decode runtime shape.
- Explicit KV-cache handoff in the runtime.
- MiniCPM chat-template and special-token constants.
- Int8 KV-cache memory budget calculation.
- SE 2 and SE 3 quantized artifact selection contracts.

Not implemented yet:

- Downloading or converting the real `openbmb/MiniCPM5-1B` checkpoint.
- Full MiniCPM tokenizer execution from `tokenizer.json`.
- Real MiniCPM prefill/decode `mlprogram` graphs.
- Real KV-cache tensor layout for 24 layers, 2 KV heads, head dimension 128.
- Stateful Core ML KV cache; current smoke path uses explicit KV input/output.
- Mixed int4/int8 quantization pipeline and PyTorch-vs-Core-ML logits validation.
- lm_head optimization, speculative decoding, thermal throttling policy, and physical SE 2/SE 3 benchmarks.
