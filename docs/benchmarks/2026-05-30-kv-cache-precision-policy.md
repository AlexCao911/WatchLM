# KV Cache Precision Policy

## Scope

This note records the policy/schema change that followed the prefill KV precision matrix.

The diagnostic showed that global int8 prefill can flip the first decode top-1 even when decode itself is fp16. The runtime and conversion contracts therefore need to express `fp16` KV cache / KV path fidelity profiles instead of assuming every deployable profile must declare `int8` KV cache.

## Change

The manifest and conversion contracts now accept:

```text
quantization.kvCache = fp16
quantization.kvCache = int8
```

`int4` KV cache remains rejected.

Added a prefill-protected mixed precision policy:

```text
tools/conversion/mixed-precision-policy-prefill-kv-protected.json
```

Its intent is:

```text
embedding: int8
lm_head: int8
norms: fp16
attention Q/K/O: fp16
attention V: fp16
FFN: int8
layer 12 FFN: int4
KV cache policy: fp16
structural reduction: false
```

## Policy Plan Evidence

Command:

```text
.venv/bin/python tools/conversion/convert-minicpm5-coreml.py --compression mixed --precision-policy tools/conversion/mixed-precision-policy-prefill-kv-protected.json --describe-compression-policy
```

Key output:

```text
policyId: prefill-kv-fp16-attn-ffn12-int4
attentionQKO: fp16
attentionV: fp16
kvCachePrecision: fp16
layerPrecision[12].ffn: int4
compressionPasses: int8, int4
```

The int8/int4 pass patterns exclude attention projections, so Q/K/O/V remain uncompressed by this policy. FFN remains compressible, with layer 12 eligible for int4.

## Verification

```text
swift test --filter MixedPrecisionPolicyTests
node --test test/modelManifest.test.js
node --test test/realConversionCli.test.js --test-name-pattern "prefill KV protected"
swift test --filter ModelManifestTests
```

All four commands passed after the schema change.

## Next Step

The next artifact experiment should recompress the existing context-16 prefill-KV graph with `mixed-precision-policy-prefill-kv-protected.json`, then run the same precision matrix:

```text
protected prefill -> int8 decode
protected prefill -> fp16 decode
```

The promotion criterion is that protected prefill restores decode top-1 `4245` on `en-short-001` while reducing size versus the full fp16 prefill graph.
