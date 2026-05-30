# Activation Importance Cal12

Date: 2026-05-30

## Scope

This note records the first full WatchLM calibration-suite activation
importance run for MiniCPM5-1B.

It is calibration evidence for policy generation, not final Core ML quality
evidence.

## Command

```bash
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --cache-dir artifacts/hf/MiniCPM5-1B \
  --top-columns 8 \
  --device cpu \
  --quiet \
  --output artifacts/benchmarks/minicpm5-activation-importance-cal12.json
```

## Output

```text
report: artifacts/benchmarks/minicpm5-activation-importance-cal12.json
mode: activation-collection
prompt count: 12
elapsed seconds: 4.925253
module count: 218
statistic: sum_input_activation_squared_by_column
```

Component summary:

```text
attentionQKO: modules=72 total=10086572.10 max=339840.13
attentionV:   modules=24 total=4086771.48  max=339840.13
ffn:          modules=72 total=67817678.33 max=44214168.00
lmHead:       modules=1  total=10848736.00 max=10848736.00
norms:        modules=49 total=8653176713.47 max=243433280.00
```

Lowest layers by linear activation total:

```text
layer0: linear=178004.37  qko=62659.39  v=30616.74  ffn=84728.23
layer1: linear=281471.04  qko=144161.32 v=71263.91  ffn=66045.80
layer2: linear=349185.89  qko=172117.46 v=85011.39  ffn=92057.04
layer3: linear=514978.65  qko=189348.02 v=92541.67  ffn=233088.96
layer6: linear=583107.88  qko=233510.73 v=107708.41 ffn=241888.74
layer5: linear=586210.17  qko=249197.77 v=119620.70 ffn=217391.70
layer7: linear=627885.75  qko=253959.45 v=116299.72 ffn=257626.57
layer9: linear=667816.39  qko=287152.74 v=125107.59 ffn=255556.05
```

Highest layers by linear activation total:

```text
layer4:  linear=44659800.93 qko=176262.18 v=83273.07  ffn=44400265.69
layer23: linear=10840156.81 qko=1013552.06 v=339840.13 ffn=9486764.63
layer22: linear=6190990.34  qko=806121.00  v=266222.53 ffn=5118646.81
layer21: linear=2060392.31  qko=832798.50  v=261841.00 ffn=965752.81
layer20: linear=1702801.34  qko=602950.28  v=273275.38 ffn=826575.69
```

## Interpretation

The full calibration suite confirms a few practical priors:

```text
norms and lm_head remain protected
FFN carries the largest linear activation energy
layer4 FFN is an outlier and should not be int4-compressed blindly
late layers 21-23 are high-energy and should stay protected for now
attention V has rankable layer-level evidence and remains the cleanest
candidate family for narrow int4 exploration
```

The lowest raw linear totals are early layers, but edge layers are already
protected by architecture policy. That means low-energy evidence should be
combined with existing priors, not applied mechanically.

## Next Candidate Direction

Generate policy candidates from low-energy linear modules while preserving:

```text
embedding
lm_head
norms
edge layers
layer4 FFN outlier
late high-energy layers 21-23
attention O until separate O-safe evidence improves
```

The first candidate generator should likely rank V-only and FFN-only windows
separately, then run each Core ML candidate through the Swift quantization
sensitivity scorer before any watch deployment claim.
