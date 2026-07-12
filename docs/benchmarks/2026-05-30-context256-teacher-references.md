# Context-256 Teacher References

## Scope

This note records the first PyTorch teacher sidecar for the SE2-oriented context-256 benchmark path.

It is separate from context-16 policy search notes. The purpose is to prevent future context-256 Core ML runs from accidentally comparing against context-16 teacher tokens.

## Command

```text
.venv/bin/python tools/benchmark/generate-teacher-references.py --context-tokens 256 --max-new-tokens 2 --output artifacts/benchmarks/minicpm5-teacher-references-context256-cap2.json
```

## Output

```text
path: artifacts/benchmarks/minicpm5-teacher-references-context256-cap2.json
promptCount: 10
contextTokens: 256
maxNewTokensCap: 2
```

Teacher prefixes:

```text
zh-short-001        [18487,45105]
zh-short-002        [242,242]
en-short-001        [1974,10300]
en-short-002        [1974,220]
code-fix-001        [35811,285]
code-fix-002        [319,370]
watch-utility-001   [354,558]
watch-utility-002   [130073]
safety-refusal-001  [1974,220]
safety-refusal-002  [1974,220]
```

## Interpretation

The context-256 teacher differs from the context-16 teacher on several prompts. For example:

```text
en-short-001 context-16:  [416,4245]
en-short-001 context-256: [1974,10300]
```

Future context-256 Core ML benchmarks must use this sidecar or regenerate a matching sidecar with the same context and max-new-token cap.

## Next

Generate context-256 Core ML prefill-KV and decode artifacts for the current best fidelity policy:

```text
prefill: attention/KV fp16, FFN int8, embedding/lm_head int8
decode: global int8
```
