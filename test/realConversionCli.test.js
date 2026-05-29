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
