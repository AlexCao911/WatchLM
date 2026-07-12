import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const python = path.join(repoRoot, ".venv", "bin", "python");
const collectorScript = path.join(repoRoot, "tools", "conversion", "collect-activation-importance.py");
const calibrationPromptsPath = path.join(
  repoRoot,
  "tools",
  "benchmark",
  "fixtures",
  "calibration-prompts.json"
);

test("activation importance CLI emits a dry-run report from calibration prompts", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-importance-"));
  const outputPath = path.join(tempDir, "importance.json");
  const { stdout } = await execFileAsync(python, [
    collectorScript,
    "--calibration-prompts",
    calibrationPromptsPath,
    "--dry-run",
    "--output",
    outputPath
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const report = JSON.parse(stdout);
  const written = JSON.parse(await readFile(outputPath, "utf8"));

  assert.deepEqual(written, report);
  assert.equal(report.schemaVersion, 1);
  assert.equal(report.sourceModelId, "openbmb/MiniCPM5-1B");
  assert.equal(report.collection.mode, "dry-run");
  assert.equal(report.collection.statistic, "sum_input_activation_squared_by_column");
  assert.equal(report.calibration.promptCount, 12);
  assert.deepEqual(report.calibration.prefixTokenCounts, [1, 2, 4, 8, 12, 18, 32]);
  assert.deepEqual(report.targetComponents, [
    "attentionQKO",
    "attentionV",
    "ffnGateUp",
    "ffnDown",
    "ffn",
    "embedding",
    "lmHead",
    "norms"
  ]);
  assert.deepEqual(report.componentSummary, []);
  assert.deepEqual(report.layerSummary, []);
  assert.deepEqual(report.modules, []);
});

test("activation importance CLI can write quietly for long real-model runs", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-importance-quiet-"));
  const outputPath = path.join(tempDir, "importance.json");
  const { stdout } = await execFileAsync(python, [
    collectorScript,
    "--calibration-prompts",
    calibrationPromptsPath,
    "--dry-run",
    "--quiet",
    "--output",
    outputPath
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const report = JSON.parse(await readFile(outputPath, "utf8"));

  assert.equal(stdout, "");
  assert.equal(report.collection.mode, "dry-run");
  assert.equal(report.calibration.promptCount, 12);
});

test("activation importance module classifier maps MiniCPM tensor families", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/collect-activation-importance.py").resolve()
spec = importlib.util.spec_from_file_location("collect_activation_importance", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

names = [
    "model.layers.11.self_attn.q_proj",
    "model.layers.11.self_attn.v_proj",
    "model.layers.11.mlp.gate_proj",
    "model.layers.11.mlp.up_proj",
    "model.layers.11.mlp.down_proj",
    "model.embed_tokens",
    "lm_head",
    "model.layers.11.input_layernorm",
]
print(json.dumps([module.classify_module_name(name) for name in names]))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

assert.deepEqual(JSON.parse(stdout), [
    "attentionQKO",
    "attentionV",
    "ffnGateUp",
    "ffnGateUp",
    "ffnDown",
    "embedding",
    "lmHead",
    "norms"
  ]);
});

test("activation importance summary aggregates modules by component and layer", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

script = Path("tools/conversion/collect-activation-importance.py").resolve()
spec = importlib.util.spec_from_file_location("collect_activation_importance", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

modules = [
    {"name": "model.layers.0.self_attn.q_proj", "component": "attentionQKO", "layerIndex": 0, "totalActivationEnergy": 10.0},
    {"name": "model.layers.0.self_attn.k_proj", "component": "attentionQKO", "layerIndex": 0, "totalActivationEnergy": 30.0},
    {"name": "model.layers.1.mlp.down_proj", "component": "ffn", "layerIndex": 1, "totalActivationEnergy": 60.0},
    {"name": "lm_head", "component": "lmHead", "layerIndex": None, "totalActivationEnergy": 100.0},
]
print(json.dumps({
    "component": module.component_summary(modules),
    "layer": module.layer_summary(modules),
}, sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const summary = JSON.parse(stdout);

  assert.deepEqual(summary.component, [
    {
      component: "attentionQKO",
      maxModuleActivationEnergy: 30,
      meanModuleActivationEnergy: 20,
      moduleCount: 2,
      totalActivationEnergy: 40
    },
    {
      component: "ffn",
      maxModuleActivationEnergy: 60,
      meanModuleActivationEnergy: 60,
      moduleCount: 1,
      totalActivationEnergy: 60
    },
    {
      component: "lmHead",
      maxModuleActivationEnergy: 100,
      meanModuleActivationEnergy: 100,
      moduleCount: 1,
      totalActivationEnergy: 100
    }
  ]);
  assert.deepEqual(summary.layer, [
    {
      componentTotals: { attentionQKO: 40 },
      layerIndex: 0,
      moduleCount: 2,
      totalActivationEnergy: 40
    },
    {
      componentTotals: { ffn: 60 },
      layerIndex: 1,
      moduleCount: 1,
      totalActivationEnergy: 60
    }
  ]);
});

test("activation importance summary reports channel group sensitivity", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
import json
from pathlib import Path

import torch

script = Path("tools/conversion/collect-activation-importance.py").resolve()
spec = importlib.util.spec_from_file_location("collect_activation_importance", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

stats = {
    "model.layers.6.self_attn.v_proj": {
        "name": "model.layers.6.self_attn.v_proj",
        "component": "attentionV",
        "layerIndex": 6,
        "observationCount": 12,
        "energy": torch.tensor([16.0, 4.0, 1.0, 1.0, 8.0, 8.0, 2.0, 0.0]),
    },
}

print(json.dumps(module.summarize_stats(stats, top_columns=3, group_size=4, top_groups=2), sort_keys=True))
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  const [moduleSummary] = JSON.parse(stdout);

  assert.equal(moduleSummary.channelGroupSize, 4);
  assert.equal(moduleSummary.channelGroupCount, 2);
  assert.deepEqual(moduleSummary.channelSummary, {
    maxColumnEnergy: 16,
    topColumnEnergyFraction: 0.4,
    topColumnsEnergyFraction: 0.8
  });
  assert.deepEqual(moduleSummary.topGroups, [
    {
      endColumnExclusive: 4,
      groupIndex: 0,
      meanActivationEnergy: 5.5,
      startColumn: 0,
      topColumnEnergy: 16,
      topColumnIndex: 0,
      totalActivationEnergy: 22
    },
    {
      endColumnExclusive: 8,
      groupIndex: 1,
      meanActivationEnergy: 4.5,
      startColumn: 4,
      topColumnEnergy: 8,
      topColumnIndex: 4,
      totalActivationEnergy: 18
    }
  ]);
});
