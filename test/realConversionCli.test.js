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
  assert.match(stdout, /--graph/);
  assert.match(stdout, /prefill/);
  assert.match(stdout, /prefill-kv/);
  assert.match(stdout, /decode/);
  assert.match(stdout, /stateful-kv/);
  assert.match(stdout, /stateful-step-kv/);
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
