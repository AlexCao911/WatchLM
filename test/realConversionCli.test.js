import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const python = path.join(repoRoot, ".venv", "bin", "python");
const conversionScript = path.join(repoRoot, "tools", "conversion", "convert-minicpm5-coreml.py");

test("real MiniCPM conversion CLI exposes compression and graph choices", async () => {
  const { stdout } = await execFileAsync(python, [conversionScript, "--help"], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.match(stdout, /--compression/);
  assert.match(stdout, /--source-mlpackage/);
  assert.match(stdout, /none/);
  assert.match(stdout, /int8/);
  assert.match(stdout, /int4/);
  assert.match(stdout, /mixed/);
  assert.match(stdout, /--precision-policy/);
  assert.match(stdout, /--torch-dtype/);
  assert.match(stdout, /--graph/);
  assert.match(stdout, /prefill/);
  assert.match(stdout, /prefill-kv/);
  assert.match(stdout, /decode/);
  assert.match(stdout, /stateful-kv/);
  assert.match(stdout, /stateful-step-kv/);
});

test("real conversion CLI resolves model load dtype choices", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
from pathlib import Path
import torch

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.resolve_torch_dtype("float16") is torch.float16
assert module.resolve_torch_dtype("float32") is torch.float32
assert module.resolve_torch_dtype("bfloat16") is torch.bfloat16
assert module.resolve_torch_dtype("auto") == "auto"
print("ok")
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.equal(stdout.trim(), "ok");
});

test("decode Core ML input types preserve past KV tensor dtype", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

config = SimpleNamespace(num_hidden_layers=1)
example_inputs = (
    torch.zeros((1, 1), dtype=torch.int32),
    torch.zeros((1, 1), dtype=torch.int32),
    torch.zeros((1, 1, 1, 17), dtype=torch.float16),
    torch.zeros((1, 8, 16, 128), dtype=torch.float32),
    torch.zeros((1, 8, 16, 128), dtype=torch.float32),
)
types = module.input_types("decode", example_inputs, config, 16)
past_key = next(item for item in types if item.name == "past_key_0")
past_value = next(item for item in types if item.name == "past_value_0")
assert past_key.dtype is not None
assert past_key.dtype.__name__ == "fp32"
assert past_value.dtype.__name__ == "fp32"
print("ok")
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.equal(stdout.trim(), "ok");
});

test("stateful KV conversion schema describes 24 Core ML state tensors", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

config = SimpleNamespace(
    num_hidden_layers=24,
    num_key_value_heads=2,
    num_attention_heads=16,
    hidden_size=1536,
    head_dim=128,
)
schema = module.stateful_kv_graph_schema(config, context_tokens=256)
state_types = module.state_types("stateful-kv", config, 256)
assert len(state_types) == 48
assert state_types[0].name == "past_key_0"
assert state_types[-1].name == "past_value_23"
attention_config = SimpleNamespace(
    num_hidden_layers=1,
    num_key_value_heads=None,
    num_attention_heads=16,
    hidden_size=1536,
)
attention_schema = module.stateful_kv_graph_schema(attention_config, context_tokens=32)
assert attention_schema["kvHeads"] == 16
assert attention_schema["headDimension"] == 96
print(json.dumps(schema, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const schema = JSON.parse(stdout);

  assert.equal(schema.interface, "stateful-kv");
  assert.equal(schema.layerCount, 24);
  assert.equal(schema.kvHeads, 2);
  assert.equal(schema.headDimension, 128);
  assert.deepEqual(schema.inputs, ["input_ids", "position_ids", "causal_mask"]);
  assert.deepEqual(schema.outputs, ["logits"]);
  assert.equal(schema.states.length, 48);
  assert.deepEqual(schema.states[0], {
    name: "past_key_0",
    shape: [1, 2, 256, 128],
    dtype: "float16"
  });
  assert.deepEqual(schema.states[1], {
    name: "past_value_0",
    shape: [1, 2, 256, 128],
    dtype: "float16"
  });
  assert.deepEqual(schema.states.at(-1), {
    name: "past_value_23",
    shape: [1, 2, 256, 128],
    dtype: "float16"
  });
});

test("stateful step KV conversion schema uses single-token IO and full state writes", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

config = SimpleNamespace(
    num_hidden_layers=24,
    num_key_value_heads=2,
    num_attention_heads=16,
    hidden_size=1536,
    head_dim=128,
)
schema = module.stateful_step_kv_graph_schema(config, context_tokens=256)
input_types = module.input_types(
    "stateful-step-kv",
    module.stateful_step_example_inputs(context_tokens=256),
    config,
    256,
)
state_types = module.state_types("stateful-step-kv", config, 256)
assert [item.name for item in input_types] == ["input_ids", "position_ids", "causal_mask"]
assert len(state_types) == 48
print(json.dumps(schema, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const schema = JSON.parse(stdout);

  assert.equal(schema.interface, "stateful-step-kv");
  assert.equal(schema.layerCount, 24);
  assert.deepEqual(schema.inputs, ["input_ids", "position_ids", "causal_mask"]);
  assert.deepEqual(schema.inputShapes, {
    input_ids: [1, 1],
    position_ids: [1, 1],
    causal_mask: [1, 1, 1, 257]
  });
  assert.equal(schema.stateUpdate, "sliding-window-full-state-write");
  assert.equal(schema.states.length, 48);
  assert.deepEqual(schema.states[0].shape, [1, 2, 256, 128]);
});

test("snapshot download patterns include single-file and sharded safetensors checkpoints", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(json.dumps(module.snapshot_allow_patterns()))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  const patterns = JSON.parse(stdout);
  assert.ok(patterns.includes("model.safetensors"));
  assert.ok(patterns.includes("model-*.safetensors"));
  assert.ok(patterns.includes("model.safetensors.index.json"));
});

test("real MiniCPM conversion CLI describes mixed precision policy without loading the model", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-policy-"));
  const policyPath = path.join(tempDir, "mixed-policy.json");
  await writeFile(policyPath, JSON.stringify({
    schemaVersion: 1,
    policyId: "mixed-int4-ffn-int8-attn-kv",
    strategy: "mixed-precision-fidelity-first",
    layerCount: 24,
    protectedEdgeLayerCount: 2,
    weights: {
      embedding: "int8",
      lmHead: "int8",
      norms: "fp16",
      attentionQKO: "int8",
      attentionV: "int8",
      ffn: "int4"
    },
    kvCache: "int8",
    structuralReduction: false,
    opNamePatterns: {
      embedding: ["embed_tokens"],
      lmHead: ["lm_head"],
      norms: ["norm"],
      attentionQKO: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"],
      attentionV: ["self_attn.v_proj"],
      ffn: ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    }
  }), "utf8");

  const { stdout } = await execFileAsync(python, [
    conversionScript,
    "--compression",
    "mixed",
    "--precision-policy",
    policyPath,
    "--describe-compression-policy"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);

  assert.equal(plan.policyId, "mixed-int4-ffn-int8-attn-kv");
  assert.equal(plan.strategy, "mixed-precision-fidelity-first");
  assert.equal(plan.layerCount, 24);
  assert.equal(plan.kvCachePrecision, "int8");
  assert.equal(plan.structuralReduction, false);
  assert.equal(plan.componentPrecision.ffn, "int4");
  assert.equal(plan.layerPrecision["0"].ffn, "int8");
  assert.equal(plan.layerPrecision["12"].ffn, "int4");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int8", "int4"]);
  assert.ok(plan.compressionPasses[0].opNamePatterns.includes("self_attn.q_proj"));
  assert.ok(plan.compressionPasses[1].opNamePatterns.includes("mlp.down_proj"));
});

test("mixed precision op selectors tolerate Core ML separator rewrites", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

assert module.pattern_matches_name("mlp.down_proj", "model_layers_12_mlp_down_proj_weight")
assert module.pattern_matches_name("self_attn.q_proj", "model.layers.0.self_attn.q_proj")
assert module.extract_layer_index("model_layers_12_mlp_down_proj_weight") == 12
assert module.extract_layer_index("model.layers.0.self_attn.q_proj") == 0
print("ok")
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.equal(stdout.trim(), "ok");
});

test("source mlpackage compression reports context inferred from source package name", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
from pathlib import Path
from types import SimpleNamespace

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

source_args = SimpleNamespace(
    context_tokens=16,
    source_mlpackage="artifacts/coreml/real-minicpm5-decode-256/decode-256.mlpackage",
)
fresh_args = SimpleNamespace(context_tokens=16, source_mlpackage=None)

assert module.reported_context_tokens(source_args) == 256
assert module.reported_context_tokens(fresh_args) == 16
print("ok")
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.equal(stdout.trim(), "ok");
});

test("mixed precision op selectors record component and layer audit evidence", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy.json")
audit = module.new_mixed_compression_audit(policy)
int8_selector = module.make_mixed_precision_op_selector(policy, "int8", audit)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int8_selector(FakeOp("model_layers_0_mlp_down_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_mlp_down_proj_weight"))
assert int8_selector(FakeOp("model_layers_12_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_12_self_attn_q_proj_weight"))
print(json.dumps(audit, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const audit = JSON.parse(stdout);

  assert.equal(audit.policyId, "mixed-int4-ffn-int8-attn-kv");
  assert.equal(audit.passes.int8.selectedOpCount, 2);
  assert.equal(audit.passes.int8.selectedByComponent.ffn, 1);
  assert.equal(audit.passes.int8.selectedByComponent.attentionQKO, 1);
  assert.equal(audit.passes.int8.selectedByLayer["0"], 1);
  assert.equal(audit.passes.int8.selectedByLayer["12"], 1);
  assert.equal(audit.passes.int4.selectedOpCount, 1);
  assert.equal(audit.passes.int4.selectedByComponent.ffn, 1);
  assert.equal(audit.passes.int4.rejectedOpCount, 1);
});

test("mixed precision policies can split FFN gate/up and down projections", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-ffn-split-policy-"));
  const policyPath = path.join(tempDir, "ffn-split-policy.json");
  await writeFile(policyPath, JSON.stringify({
    schemaVersion: 1,
    policyId: "mixed-split-ffn-gateup-down",
    strategy: "mixed-precision-fidelity-first",
    layerCount: 24,
    protectedEdgeLayerCount: 0,
    weights: {
      embedding: "fp16",
      lmHead: "fp16",
      norms: "fp16",
      attentionQKO: "fp16",
      attentionV: "fp16",
      ffn: "fp16",
      ffnGateUp: "int4",
      ffnDown: "int8"
    },
    layerOverrides: {
      ffnGateUp: {
        "12": "fp16"
      },
      ffnDown: {
        "12": "int4"
      }
    },
    kvCache: "fp16",
    structuralReduction: false,
    opNamePatterns: {
      ffnGateUp: ["mlp.gate_proj", "mlp.up_proj"],
      ffnDown: ["mlp.down_proj"]
    }
  }), "utf8");

  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(${JSON.stringify(policyPath)})
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int8_selector = module.make_mixed_precision_op_selector(policy, "int8", audit)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_11_mlp_gate_proj_weight"))
assert int4_selector(FakeOp("model_layers_11_mlp_up_proj_weight"))
assert not int4_selector(FakeOp("model_layers_11_mlp_down_proj_weight"))
assert int8_selector(FakeOp("model_layers_11_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_12_mlp_gate_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_mlp_down_proj_weight"))

print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);

  assert.equal(result.plan.policyId, "mixed-split-ffn-gateup-down");
  assert.equal(result.plan.componentPrecision.ffn, "fp16");
  assert.equal(result.plan.componentPrecision.ffnGateUp, "int4");
  assert.equal(result.plan.componentPrecision.ffnDown, "int8");
  assert.equal(result.plan.layerPrecision["11"].ffnGateUp, "int4");
  assert.equal(result.plan.layerPrecision["11"].ffnDown, "int8");
  assert.equal(result.plan.layerPrecision["12"].ffnGateUp, "fp16");
  assert.equal(result.plan.layerPrecision["12"].ffnDown, "int4");
  assert.equal(result.audit.passes.int4.selectedByComponent.ffnGateUp, 2);
  assert.equal(result.audit.passes.int4.selectedByComponent.ffnDown, 1);
  assert.equal(result.audit.passes.int8.selectedByComponent.ffnDown, 1);
});

test("mixed precision policies can restrict int4 to explicit layer overrides", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-policy-"));
  const policyPath = path.join(tempDir, "ffn12-policy.json");
  await writeFile(policyPath, JSON.stringify({
    schemaVersion: 1,
    policyId: "mixed-int4-ffn12-int8-rest",
    strategy: "mixed-precision-fidelity-first",
    layerCount: 24,
    protectedEdgeLayerCount: 2,
    weights: {
      embedding: "int8",
      lmHead: "int8",
      norms: "fp16",
      attentionQKO: "int8",
      attentionV: "int8",
      ffn: "int8"
    },
    layerOverrides: {
      ffn: {
        "12": "int4"
      }
    },
    kvCache: "int8",
    structuralReduction: false,
    opNamePatterns: {
      embedding: ["embed_tokens"],
      lmHead: ["lm_head"],
      norms: ["norm"],
      attentionQKO: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"],
      attentionV: ["self_attn.v_proj"],
      ffn: ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    }
  }), "utf8");

  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(${JSON.stringify(policyPath)})
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int8_selector = module.make_mixed_precision_op_selector(policy, "int8", audit)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int8_selector(FakeOp("model_layers_11_mlp_down_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_mlp_down_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);

  assert.equal(result.plan.policyId, "mixed-int4-ffn12-int8-rest");
  assert.equal(result.plan.layerPrecision["11"].ffn, "int8");
  assert.equal(result.plan.layerPrecision["12"].ffn, "int4");
  assert.equal(result.plan.layerPrecision["13"].ffn, "int8");
  assert.equal(result.plan.compressionPasses.find((pass) => pass.precision === "int4").opNamePatterns.length, 3);
  assert.equal(result.audit.passes.int4.selectedOpCount, 1);
  assert.deepEqual(result.audit.passes.int4.selectedByLayer, { "12": 1 });
});

test("mixed precision policies can describe grouped-channel int4 palettization", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-policy-"));
  const policyPath = path.join(tempDir, "grouped-channel-policy.json");
  await writeFile(policyPath, JSON.stringify({
    schemaVersion: 1,
    policyId: "stateful-step-attention-int4-grouped-channel",
    strategy: "mixed-precision-fidelity-first",
    layerCount: 24,
    protectedEdgeLayerCount: 0,
    weights: {
      embedding: "fp16",
      lmHead: "fp16",
      norms: "fp16",
      attentionQKO: "fp16",
      attentionV: "fp16",
      ffn: "fp16"
    },
    layerOverrides: {
      attentionQKO: {
        "11": "int4"
      },
      attentionV: {
        "11": "int4"
      }
    },
    int4Compression: {
      method: "palettization",
      mode: "kmeans",
      granularity: "per_grouped_channel",
      groupSize: 16,
      enablePerChannelScale: true,
      clusterDim: 1,
      numKMeansWorkers: 1,
      weightThreshold: 2048
    },
    kvCache: "fp16",
    structuralReduction: false
  }), "utf8");

  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

policy = module.load_mixed_precision_policy(${JSON.stringify(policyPath)})
plan = module.build_mixed_compression_plan(policy)
print(json.dumps(plan["compressionPasses"], sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const passes = JSON.parse(stdout);
  const int4Pass = passes.find((pass) => pass.precision === "int4");

  assert.equal(int4Pass.method, "kmeans_palettization");
  assert.deepEqual(int4Pass.settings, {
    clusterDim: 1,
    enablePerChannelScale: true,
    granularity: "per_grouped_channel",
    groupSize: 16,
    method: "palettization",
    mode: "kmeans",
    numKMeansWorkers: 1,
    weightThreshold: 2048
  });
});

test("prefill KV protected policy keeps attention and KV cache at fp16", async () => {
  const { stdout } = await execFileAsync(python, [
    conversionScript,
    "--compression",
    "mixed",
    "--precision-policy",
    "tools/conversion/mixed-precision-policy-prefill-kv-protected.json",
    "--describe-compression-policy"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);
  const int8Pass = plan.compressionPasses.find((pass) => pass.precision === "int8");
  const int4Pass = plan.compressionPasses.find((pass) => pass.precision === "int4");

  assert.equal(plan.policyId, "prefill-kv-fp16-attn-ffn12-int4");
  assert.equal(plan.kvCachePrecision, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.layerPrecision["12"].ffn, "int4");
  assert.ok(int8Pass.opNamePatterns.includes("lm_head"));
  assert.ok(!int8Pass.opNamePatterns.includes("self_attn.q_proj"));
  assert.ok(!int8Pass.opNamePatterns.includes("self_attn.v_proj"));
  assert.ok(int4Pass.opNamePatterns.includes("mlp.down_proj"));
  assert.ok(int4Pass.opNamePatterns.includes("mlp.gate_proj"));
  assert.ok(int4Pass.opNamePatterns.includes("mlp.up_proj"));
  assert.ok(!int4Pass.opNamePatterns.includes("self_attn.q_proj"));
  assert.ok(!int4Pass.opNamePatterns.includes("self_attn.v_proj"));
});

test("prefill KV protected no-int4 policy emits only the int8 compression pass", async () => {
  const { stdout } = await execFileAsync(python, [
    conversionScript,
    "--compression",
    "mixed",
    "--precision-policy",
    "tools/conversion/mixed-precision-policy-prefill-kv-protected-no-int4.json",
    "--describe-compression-policy"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);

  assert.equal(plan.policyId, "prefill-kv-fp16-attn-ffn-int8");
  assert.equal(plan.kvCachePrecision, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.layerPrecision["12"].ffn, "int8");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int8"]);
  assert.ok(plan.compressionPasses[0].opNamePatterns.includes("mlp.down_proj"));
  assert.ok(!plan.compressionPasses[0].opNamePatterns.includes("self_attn.q_proj"));
  assert.ok(!plan.compressionPasses[0].opNamePatterns.includes("self_attn.v_proj"));
});

test("stateful step protected no-int4 policy keeps attention and KV state at fp16", async () => {
  const { stdout } = await execFileAsync(python, [
    conversionScript,
    "--compression",
    "mixed",
    "--precision-policy",
    "tools/conversion/mixed-precision-policy-stateful-step-protected-no-int4.json",
    "--describe-compression-policy"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);

  assert.equal(plan.policyId, "stateful-step-fp16-attn-ffn-int8");
  assert.equal(plan.kvCachePrecision, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "int8");
  assert.equal(plan.layerPrecision["0"].ffn, "int8");
  assert.equal(plan.layerPrecision["12"].ffn, "int8");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int8"]);
  assert.ok(plan.compressionPasses[0].opNamePatterns.includes("lm_head"));
  assert.ok(plan.compressionPasses[0].opNamePatterns.includes("mlp.down_proj"));
  assert.ok(!plan.compressionPasses[0].opNamePatterns.includes("self_attn.q_proj"));
  assert.ok(!plan.compressionPasses[0].opNamePatterns.includes("self_attn.v_proj"));
});

test("stateful step FFN-only int8 policy keeps embedding and lm head at fp16", async () => {
  const { stdout } = await execFileAsync(python, [
    conversionScript,
    "--compression",
    "mixed",
    "--precision-policy",
    "tools/conversion/mixed-precision-policy-stateful-step-ffn-int8.json",
    "--describe-compression-policy"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);

  assert.equal(plan.policyId, "stateful-step-fp16-embed-lmhead-attn-ffn-int8");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "int8");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int8"]);
  assert.deepEqual(plan.compressionPasses[0].opNamePatterns, [
    "feed_forward",
    "ffn",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj"
  ]);
});

test("stateful step single-layer FFN int8 policy leaves most FFN layers at fp16", async () => {
  const { stdout } = await execFileAsync(python, [
    conversionScript,
    "--compression",
    "mixed",
    "--precision-policy",
    "tools/conversion/mixed-precision-policy-stateful-step-ffn12-int8.json",
    "--describe-compression-policy"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);

  assert.equal(plan.policyId, "stateful-step-ffn12-int8-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["11"].ffn, "fp16");
  assert.equal(plan.layerPrecision["12"].ffn, "int8");
  assert.equal(plan.layerPrecision["13"].ffn, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int8"]);
  assert.deepEqual(plan.compressionPasses[0].opNamePatterns, [
    "feed_forward",
    "ffn",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj"
  ]);
});

test("stateful step early-layer int4 policy protects embeddings and first four layers", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-early4-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert not int4_selector(FakeOp("model_model_embed_tokens_weight"))
assert not int4_selector(FakeOp("model_model_lm_head_weight"))
assert not int4_selector(FakeOp("model_layers_3_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_3_self_attn_q_proj_weight"))
assert int4_selector(FakeOp("model_layers_4_mlp_down_proj_weight"))
assert int4_selector(FakeOp("model_layers_4_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-fp16-embed-lmhead-early4-rest-int4");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "int4");
  assert.equal(plan.componentPrecision.attentionV, "int4");
  assert.equal(plan.componentPrecision.ffn, "int4");
  assert.equal(plan.layerPrecision["0"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["0"].ffn, "fp16");
  assert.equal(plan.layerPrecision["3"].attentionV, "fp16");
  assert.equal(plan.layerPrecision["3"].ffn, "fp16");
  assert.equal(plan.layerPrecision["4"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["4"].ffn, "int4");
  assert.equal(plan.layerPrecision["23"].attentionQKO, "int4");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 2);
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "4": 2 });
});

test("stateful step layer23-only int4 policy isolates a single late layer", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer23-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert not int4_selector(FakeOp("model_layers_22_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_22_self_attn_q_proj_weight"))
assert int4_selector(FakeOp("model_layers_23_mlp_down_proj_weight"))
assert int4_selector(FakeOp("model_layers_23_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer23-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["22"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["22"].ffn, "fp16");
  assert.equal(plan.layerPrecision["23"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["23"].attentionV, "int4");
  assert.equal(plan.layerPrecision["23"].ffn, "int4");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 2);
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "23": 2 });
});

test("stateful step layer0-only int4 policy isolates the first layer", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer0-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_0_mlp_down_proj_weight"))
assert int4_selector(FakeOp("model_layers_0_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_1_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_23_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer0-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["0"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["0"].attentionV, "int4");
  assert.equal(plan.layerPrecision["0"].ffn, "int4");
  assert.equal(plan.layerPrecision["1"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["23"].ffn, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 2);
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "0": 2 });
});

test("stateful step layer0 attention-only int4 policy keeps FFN fp16", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer0-attention-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_0_self_attn_q_proj_weight"))
assert int4_selector(FakeOp("model_layers_0_self_attn_k_proj_weight"))
assert int4_selector(FakeOp("model_layers_0_self_attn_v_proj_weight"))
assert int4_selector(FakeOp("model_layers_0_self_attn_o_proj_weight"))
assert not int4_selector(FakeOp("model_layers_0_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_1_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer0-attention-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["0"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["0"].attentionV, "int4");
  assert.equal(plan.layerPrecision["0"].ffn, "fp16");
  assert.equal(plan.layerPrecision["1"].attentionQKO, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 4);
  assert.deepEqual(audit.passes.int4.selectedByComponent, {
    attentionQKO: 3,
    attentionV: 1
  });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "0": 4 });
});

test("stateful step layer0 FFN-only int4 policy keeps attention fp16", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer0-ffn-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_0_mlp_gate_proj_weight"))
assert int4_selector(FakeOp("model_layers_0_mlp_up_proj_weight"))
assert int4_selector(FakeOp("model_layers_0_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_0_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_0_self_attn_v_proj_weight"))
assert not int4_selector(FakeOp("model_layers_1_mlp_down_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer0-ffn-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["0"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["0"].attentionV, "fp16");
  assert.equal(plan.layerPrecision["0"].ffn, "int4");
  assert.equal(plan.layerPrecision["1"].ffn, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 3);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { ffn: 3 });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "0": 3 });
});

test("stateful step layer12 FFN-only int4 policy isolates a middle FFN layer", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer12-ffn-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_12_mlp_gate_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_mlp_up_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_12_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_11_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_mlp_down_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer12-ffn-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["12"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["12"].attentionV, "fp16");
  assert.equal(plan.layerPrecision["12"].ffn, "int4");
  assert.equal(plan.layerPrecision["11"].ffn, "fp16");
  assert.equal(plan.layerPrecision["13"].ffn, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 3);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { ffn: 3 });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "12": 3 });
});

test("stateful step layer12 attention-only int4 policy keeps FFN fp16", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer12-attention-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_12_self_attn_q_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_self_attn_k_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_self_attn_v_proj_weight"))
assert int4_selector(FakeOp("model_layers_12_self_attn_o_proj_weight"))
assert not int4_selector(FakeOp("model_layers_12_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_11_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer12-attention-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["12"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["12"].attentionV, "int4");
  assert.equal(plan.layerPrecision["12"].ffn, "fp16");
  assert.equal(plan.layerPrecision["11"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["13"].attentionQKO, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 4);
  assert.deepEqual(audit.passes.int4.selectedByComponent, {
    attentionQKO: 3,
    attentionV: 1
  });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "12": 4 });
});

test("stateful step layer10-13 attention-only int4 policy widens the stable middle attention window", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer10-13-attention-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(10, 14):
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_9_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_14_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer10-13-attention-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  for (const layer of ["10", "11", "12", "13"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "int4");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(plan.layerPrecision["9"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["14"].attentionQKO, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 16);
  assert.deepEqual(audit.passes.int4.selectedByComponent, {
    attentionQKO: 12,
    attentionV: 4
  });
  assert.deepEqual(audit.passes.int4.selectedByLayer, {
    "10": 4,
    "11": 4,
    "12": 4,
    "13": 4
  });
});

test("stateful step layer11-12 attention-only int4 policy narrows the middle attention window", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(11, 13):
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_10_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  for (const layer of ["11", "12"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "int4");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(plan.layerPrecision["10"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["13"].attentionQKO, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 8);
  assert.deepEqual(audit.passes.int4.selectedByComponent, {
    attentionQKO: 6,
    attentionV: 2
  });
  assert.deepEqual(audit.passes.int4.selectedByLayer, {
    "11": 4,
    "12": 4
  });
});

test("stateful step layer11-12 grouped-channel attention int4 policy narrows compression error", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-int4-grouped-channel.json"
)
plan = module.build_mixed_compression_plan(policy)
print(json.dumps(plan, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);
  const int4Pass = plan.compressionPasses.find((pass) => pass.precision === "int4");

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-int4-grouped-channel-rest-fp16");
  assert.equal(plan.layerPrecision["11"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["12"].attentionV, "int4");
  assert.equal(plan.layerPrecision["10"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["13"].attentionV, "fp16");
  assert.equal(int4Pass.settings.granularity, "per_grouped_channel");
  assert.equal(int4Pass.settings.groupSize, 16);
  assert.equal(int4Pass.settings.enablePerChannelScale, true);
});

test("stateful step layer11-12 grouped-channel no-scale policy isolates compiler-compatible grouped LUT", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-int4-grouped-channel-noscale.json"
)
plan = module.build_mixed_compression_plan(policy)
print(json.dumps(plan, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const plan = JSON.parse(stdout);
  const int4Pass = plan.compressionPasses.find((pass) => pass.precision === "int4");

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-int4-grouped-channel-noscale-rest-fp16");
  assert.equal(plan.layerPrecision["11"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["12"].attentionV, "int4");
  assert.equal(int4Pass.settings.granularity, "per_grouped_channel");
  assert.equal(int4Pass.settings.groupSize, 16);
  assert.equal(int4Pass.settings.enablePerChannelScale, false);
});

test("stateful step layer11-12 QKO-only attention int4 policy isolates attention-score projections", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-qko-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(11, 13):
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_10_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-qko-int4-rest-fp16");
  for (const layer of ["11", "12"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "int4");
    assert.equal(plan.layerPrecision[layer].attentionV, "fp16");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(audit.passes.int4.selectedOpCount, 6);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { attentionQKO: 6 });
});

test("stateful step layer11-12 QK-only attention int4 policy isolates attention-score projections", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-qk-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(11, 13):
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_10_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;
  const int4Pass = plan.compressionPasses.find((pass) => pass.precision === "int4");

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-qk-int4-rest-fp16");
  for (const layer of ["11", "12"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "int4");
    assert.equal(plan.layerPrecision[layer].attentionV, "fp16");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.deepEqual(int4Pass.opNamePatterns, [
    "attention.wk",
    "attention.wq",
    "self_attn.k_proj",
    "self_attn.q_proj"
  ]);
  assert.equal(audit.passes.int4.selectedOpCount, 4);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { attentionQKO: 4 });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { 11: 2, 12: 2 });
});

test("stateful step layer11-12 O-only attention int4 policy isolates output projection", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-o-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(11, 13):
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_10_self_attn_o_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_o_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;
  const int4Pass = plan.compressionPasses.find((pass) => pass.precision === "int4");

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-o-int4-rest-fp16");
  for (const layer of ["11", "12"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "int4");
    assert.equal(plan.layerPrecision[layer].attentionV, "fp16");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.deepEqual(int4Pass.opNamePatterns, [
    "attention.wo",
    "self_attn.o_proj"
  ]);
  assert.equal(audit.passes.int4.selectedOpCount, 2);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { attentionQKO: 2 });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { 11: 1, 12: 1 });
});

test("stateful step layer11-12 V-only attention int4 policy isolates value projections", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer11-12-attention-v-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(11, 13):
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_10_self_attn_v_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_v_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer11-12-attention-v-int4-rest-fp16");
  for (const layer of ["11", "12"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "fp16");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(audit.passes.int4.selectedOpCount, 2);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { attentionV: 2 });
});

test("stateful step layer10-13 V-only attention int4 policy widens the safe value axis", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer10-13-attention-v-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(10, 14):
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_9_self_attn_v_proj_weight"))
assert not int4_selector(FakeOp("model_layers_14_self_attn_v_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer10-13-attention-v-int4-rest-fp16");
  for (const layer of ["10", "11", "12", "13"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "fp16");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(plan.layerPrecision["9"].attentionV, "fp16");
  assert.equal(plan.layerPrecision["14"].attentionV, "fp16");
  assert.equal(audit.passes.int4.selectedOpCount, 4);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { attentionV: 4 });
  assert.deepEqual(audit.passes.int4.selectedByLayer, {
    "10": 1,
    "11": 1,
    "12": 1,
    "13": 1
  });
});

test("stateful step layer8-15 V-only attention int4 policy widens the middle value band", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer8-15-attention-v-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(8, 16):
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

assert not int4_selector(FakeOp("model_layers_7_self_attn_v_proj_weight"))
assert not int4_selector(FakeOp("model_layers_16_self_attn_v_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer8-15-attention-v-int4-rest-fp16");
  for (const layer of ["8", "9", "10", "11", "12", "13", "14", "15"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "fp16");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(plan.layerPrecision["7"].attentionV, "fp16");
  assert.equal(plan.layerPrecision["16"].attentionV, "fp16");
  assert.equal(audit.passes.int4.selectedOpCount, 8);
  assert.deepEqual(audit.passes.int4.selectedByComponent, { attentionV: 8 });
  assert.deepEqual(audit.passes.int4.selectedByLayer, {
    "8": 1,
    "9": 1,
    "10": 1,
    "11": 1,
    "12": 1,
    "13": 1,
    "14": 1,
    "15": 1
  });
});

test("stateful step layer8-15 V plus layer11-12 QK int4 policy composes safe attention axes", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy(
    "tools/conversion/mixed-precision-policy-stateful-step-layer8-15-v-layer11-12-qk-int4.json"
)
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

for layer in range(8, 16):
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_v_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_o_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_mlp_down_proj_weight"))

for layer in range(11, 13):
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))

for layer in [8, 9, 10, 13, 14, 15]:
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_q_proj_weight"))
    assert not int4_selector(FakeOp(f"model_layers_{layer}_self_attn_k_proj_weight"))

assert not int4_selector(FakeOp("model_layers_7_self_attn_v_proj_weight"))
assert not int4_selector(FakeOp("model_layers_16_self_attn_v_proj_weight"))
assert not int4_selector(FakeOp("model_layers_10_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_13_self_attn_k_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;
  const int4Pass = plan.compressionPasses.find((pass) => pass.precision === "int4");

  assert.equal(plan.policyId, "stateful-step-layer8-15-v-layer11-12-qk-int4-rest-fp16");
  assert.deepEqual(int4Pass.opNamePatterns, [
    "attention.wk",
    "attention.wq",
    "attention.wv",
    "self_attn.k_proj",
    "self_attn.q_proj",
    "self_attn.v_proj"
  ]);
  for (const layer of ["8", "9", "10", "13", "14", "15"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "fp16");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  for (const layer of ["11", "12"]) {
    assert.equal(plan.layerPrecision[layer].attentionQKO, "int4");
    assert.equal(plan.layerPrecision[layer].attentionV, "int4");
    assert.equal(plan.layerPrecision[layer].ffn, "fp16");
  }
  assert.equal(plan.layerPrecision["7"].attentionV, "fp16");
  assert.equal(plan.layerPrecision["16"].attentionV, "fp16");
  assert.equal(audit.passes.int4.selectedOpCount, 12);
  assert.deepEqual(audit.passes.int4.selectedByComponent, {
    attentionQKO: 4,
    attentionV: 8
  });
  assert.deepEqual(audit.passes.int4.selectedByLayer, {
    "8": 1,
    "9": 1,
    "10": 1,
    "11": 3,
    "12": 3,
    "13": 1,
    "14": 1,
    "15": 1
  });
});

test("stateful step layer11 attention-only int4 policy tests the left neighbor of layer12", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/convert-minicpm5-coreml.py").resolve()
spec = importlib.util.spec_from_file_location("convert_minicpm5_coreml", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeOp:
    def __init__(self, name):
        self.name = name

policy = module.load_mixed_precision_policy("tools/conversion/mixed-precision-policy-stateful-step-layer11-attention-int4.json")
plan = module.build_mixed_compression_plan(policy)
audit = module.new_mixed_compression_audit(policy)
int4_selector = module.make_mixed_precision_op_selector(policy, "int4", audit)

assert int4_selector(FakeOp("model_layers_11_self_attn_q_proj_weight"))
assert int4_selector(FakeOp("model_layers_11_self_attn_k_proj_weight"))
assert int4_selector(FakeOp("model_layers_11_self_attn_v_proj_weight"))
assert int4_selector(FakeOp("model_layers_11_self_attn_o_proj_weight"))
assert not int4_selector(FakeOp("model_layers_11_mlp_down_proj_weight"))
assert not int4_selector(FakeOp("model_layers_10_self_attn_q_proj_weight"))
assert not int4_selector(FakeOp("model_layers_12_self_attn_q_proj_weight"))
print(json.dumps({"plan": plan, "audit": audit}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const result = JSON.parse(stdout);
  const { plan, audit } = result;

  assert.equal(plan.policyId, "stateful-step-layer11-attention-int4-rest-fp16");
  assert.equal(plan.componentPrecision.embedding, "fp16");
  assert.equal(plan.componentPrecision.lmHead, "fp16");
  assert.equal(plan.componentPrecision.attentionQKO, "fp16");
  assert.equal(plan.componentPrecision.attentionV, "fp16");
  assert.equal(plan.componentPrecision.ffn, "fp16");
  assert.equal(plan.layerPrecision["11"].attentionQKO, "int4");
  assert.equal(plan.layerPrecision["11"].attentionV, "int4");
  assert.equal(plan.layerPrecision["11"].ffn, "fp16");
  assert.equal(plan.layerPrecision["10"].attentionQKO, "fp16");
  assert.equal(plan.layerPrecision["12"].attentionQKO, "fp16");
  assert.deepEqual(plan.compressionPasses.map((pass) => pass.precision), ["int4"]);
  assert.equal(audit.passes.int4.selectedOpCount, 4);
  assert.deepEqual(audit.passes.int4.selectedByComponent, {
    attentionQKO: 3,
    attentionV: 1
  });
  assert.deepEqual(audit.passes.int4.selectedByLayer, { "11": 4 });
});

test("real MiniCPM conversion CLI can reject source package compression without a compression mode", async () => {
  await assert.rejects(
    execFileAsync(python, [
      conversionScript,
      "--source-mlpackage",
      "artifacts/coreml/real-minicpm5-decode-16/decode-16.mlpackage",
      "--compression",
      "none"
    ], {
      cwd: repoRoot,
      maxBuffer: 1024 * 1024
    }),
    /--source-mlpackage requires --compression int8, int4, or mixed/
  );
});
