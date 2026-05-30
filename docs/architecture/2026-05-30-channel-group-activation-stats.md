# Channel Group Activation Statistics

Date: 2026-05-30

## Scope

This note records the first collector upgrade after the community evidence
refresh. The goal is to move from module-level activation energy toward
channel/group-level sensitivity data that can support AWQ/imatrix-style policy
generation.

## Implementation

`tools/conversion/collect-activation-importance.py` now accepts:

```text
--group-size N
--top-groups N
```

Each module entry now includes:

```text
channelGroupSize
channelGroupCount
channelSummary.maxColumnEnergy
channelSummary.topColumnEnergyFraction
channelSummary.topColumnsEnergyFraction
topGroups[]
```

`topGroups[]` records the highest-energy input-channel groups:

```text
groupIndex
startColumn
endColumnExclusive
totalActivationEnergy
meanActivationEnergy
topColumnIndex
topColumnEnergy
```

This keeps the report compatible with the previous module/layer/component
summary while adding the data needed for group-aware policy search.

## TDD

Red test:

```text
node --test test/activationImportanceCli.test.js
```

Expected failure:

```text
TypeError: summarize_stats() got an unexpected keyword argument 'group_size'
```

Green test:

```text
node --test test/activationImportanceCli.test.js
```

Result:

```text
tests 5
pass 5
fail 0
```

The fixture uses a small activation vector:

```text
[16, 4, 1, 1, 8, 8, 2, 0]
```

and verifies:

```text
group size:                  4
group count:                 2
top column energy fraction:  0.4
top 3 columns energy share:  0.8
top groups:                  columns 0-3, then columns 4-7
```

## Real Calibration Run

Command:

```text
.venv/bin/python tools/conversion/collect-activation-importance.py \
  --calibration-prompts tools/benchmark/fixtures/calibration-prompts.json \
  --cache-dir artifacts/hf/MiniCPM5-1B \
  --top-columns 8 \
  --group-size 32 \
  --top-groups 4 \
  --device cpu \
  --quiet \
  --output artifacts/benchmarks/minicpm5-activation-importance-cal12-groups.json
```

Result:

```text
prompt count:       12
module count:       218
elapsed seconds:    6.360385
group size:         32
top groups/module:  4
```

Sample module:

```text
model.layers.6.self_attn.v_proj
total activation energy:     107708.40625
max column energy:           3954.75098
top column energy fraction:  0.0367
top 8 columns energy share:  0.0643

highest group:
  columns:                   0-31
  total activation energy:   5640.9165
  top column:                25
  top column energy:         3954.7510
```

## Early Signal

The grouped report gives a sharper explanation for local failures:

```text
model.layers.0.mlp.down_proj
total activation energy:     45031.30078
top column energy fraction:  0.4341
highest group columns:       3776-3807
highest group total energy:  19566.56836
```

That single-channel concentration is a useful warning. It suggests early FFN
down-projection failures are not just "the layer is sensitive"; they may be
driven by narrow channel outliers that a per-tensor int4 LUT handles badly.

## Consequence

The next policy generator should not only ask:

```text
which tensor family has low total activation energy?
```

It should also ask:

```text
does this tensor have concentrated high-energy channels that require int8/fp16
protection or grouped-channel quantization?
```

This makes the next mixed policy more principled:

```text
int4: low-energy, low-concentration tensor groups
int8: medium-energy or concentrated groups
fp16: embeddings, lm_head, norms, Q/K/O until proven safe, and extreme outliers
```
