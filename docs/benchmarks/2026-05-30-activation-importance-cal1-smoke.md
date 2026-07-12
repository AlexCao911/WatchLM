# Activation Importance Cal1 Smoke

Date: 2026-05-30

## Scope

This note records the first real MiniCPM5-1B activation-importance collector
run.

It is a smoke run, not a final quantization policy decision. The purpose is to
prove that the collector can load local weights, run a calibration prompt, and
write module-level activation statistics.

## Command

```bash
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --cache-dir artifacts/hf/MiniCPM5-1B \
  --max-prompts 1 \
  --top-columns 8 \
  --device cpu \
  --output artifacts/benchmarks/minicpm5-activation-importance-cal1.json
```

## Output

```text
report: artifacts/benchmarks/minicpm5-activation-importance-cal1.json
mode: activation-collection
prompt count: 1
elapsed seconds: 2.198879
module count: 218
statistic: sum_input_activation_squared_by_column
```

Component summary:

```text
attentionQKO: modules=72 total=852883.57 max=28712.43
attentionV:   modules=24 total=345891.55 max=27978.28
ffn:          modules=72 total=5688757.68 max=3684509.25
lmHead:       modules=1  total=1104386.63 max=1104386.63
norms:        modules=49 total=721975011.14 max=20325702.00
```

Top layers by total activation energy were dominated by norm modules:

```text
layer19: 40729183.94
layer20: 40698942.94
layer22: 40629510.63
layer21: 40487339.88
layer18: 40466512.38
```

## Interpretation

The collector is now proven on the real local model path. It loaded the 2.0 GB
MiniCPM5 safetensors checkpoint and collected activation statistics without
network access or Core ML conversion.

This single-prompt report should not be used to select a final policy. It does
show that component families now have measurable, comparable evidence within
their own family:

```text
attention Q/K/O can be ranked by layer/module
attention V can be ranked separately
FFN can be ranked by layer/module
lm_head and norms should remain protected
```

`norms` have a much larger activation scale, so they should remain a protected
component rather than being used to normalize or promote int4 candidates.

## Next Step

Run the same collector across the full 12-prompt calibration suite, then
generate a small set of policy candidates from low-sensitivity linear modules
and gate those candidates with the Swift quantization sensitivity scorer.
