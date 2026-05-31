import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const python = path.join(repoRoot, ".venv", "bin", "python");
const scorerScript = path.join(repoRoot, "tools", "conversion", "score-quantization-risk.py");

test("quantization risk scorer gates low-bit modules by activation-weighted risk", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-risk-scorer-"));
  const reportPath = path.join(tempDir, "importance.json");
  const policyPath = path.join(tempDir, "policy.json");
  const outputPath = path.join(tempDir, "risk.json");
  await writeFile(reportPath, JSON.stringify(sampleImportanceReport()), "utf8");
  await writeFile(policyPath, JSON.stringify(samplePolicy()), "utf8");

  const { stdout } = await execFileAsync(python, [
    scorerScript,
    "--importance-report",
    reportPath,
    "--precision-policy",
    policyPath,
    "--max-weighted-risk",
    "0.2",
    "--max-top-column-fraction",
    "0.1",
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
  assert.equal(report.policyId, "sample-mixed-risk-policy");
  assert.equal(report.summary.scoredModuleCount, 2);
  assert.equal(report.summary.rejectedModuleCount, 1);
  assert.equal(report.gate.ok, false);
  assert.deepEqual(report.gate.failures, [
    "model.layers.2.self_attn.v_proj weightedRiskScore 0.409 exceeds 0.200",
    "model.layers.2.self_attn.v_proj topColumnEnergyFraction 0.120 exceeds 0.100"
  ]);
  assert.deepEqual(report.modules.map((item) => ({
    name: item.name,
    precision: item.precision,
    weightedRiskScore: item.weightedRiskScore,
    topColumnEnergyFraction: item.topColumnEnergyFraction,
    topGroupEnergyFraction: item.topGroupEnergyFraction,
    rejected: item.rejected
  })), [
    {
      name: "model.layers.2.self_attn.v_proj",
      precision: "int8",
      weightedRiskScore: 0.409,
      topColumnEnergyFraction: 0.12,
      topGroupEnergyFraction: 0.4,
      rejected: true
    },
    {
      name: "model.layers.1.self_attn.v_proj",
      precision: "int4",
      weightedRiskScore: 0.096,
      topColumnEnergyFraction: 0.05,
      topGroupEnergyFraction: 0.2,
      rejected: false
    }
  ]);
});

test("quantization risk scorer gates same-layer accumulated low-bit risk", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-risk-layer-"));
  const reportPath = path.join(tempDir, "importance.json");
  const policyPath = path.join(tempDir, "policy.json");
  await writeFile(reportPath, JSON.stringify(sampleLayerRiskImportanceReport()), "utf8");
  await writeFile(policyPath, JSON.stringify(sampleLayerRiskPolicy()), "utf8");

  const { stdout } = await execFileAsync(python, [
    scorerScript,
    "--importance-report",
    reportPath,
    "--precision-policy",
    policyPath,
    "--max-weighted-risk",
    "0.11",
    "--max-layer-weighted-risk",
    "0.15"
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const report = JSON.parse(stdout);

  assert.equal(report.summary.scoredModuleCount, 2);
  assert.equal(report.summary.rejectedLayerCount, 1);
  assert.deepEqual(report.gate.failures, [
    "layer 1 weightedRiskScore 0.200 exceeds 0.150"
  ]);
  assert.deepEqual(report.layerSummary, [
    {
      layerIndex: 1,
      scoredModuleCount: 2,
      weightedRiskScore: 0.2,
      rejected: true
    }
  ]);
  assert.equal(report.modules.every((item) => item.rejected === false), true);
});

function sampleImportanceReport() {
  return {
    schemaVersion: 1,
    sourceModelId: "openbmb/MiniCPM5-1B",
    calibration: {
      promptCount: 12,
      contextTokens: 256,
      prefixTokenCounts: [1, 2, 4, 8, 12, 18, 32]
    },
    modules: [
      module("model.layers.1.self_attn.v_proj", "attentionV", 1, 10, 0.05, 2),
      module("model.layers.2.self_attn.v_proj", "attentionV", 2, 100, 0.12, 40),
      module("model.layers.3.self_attn.v_proj", "attentionV", 3, 20, 0.01, 3),
      module("model.layers.2.self_attn.q_proj", "attentionQKO", 2, 50, 0.2, 25)
    ]
  };
}

function samplePolicy() {
  return {
    schemaVersion: 1,
    policyId: "sample-mixed-risk-policy",
    layerCount: 4,
    weights: {
      attentionV: "fp16",
      attentionQKO: "fp16",
      ffn: "fp16",
      embedding: "fp16",
      lmHead: "fp16",
      norms: "fp16"
    },
    layerOverrides: {
      attentionV: {
        1: "int4",
        2: "int8",
        3: "fp16"
      }
    },
    kvCache: "fp16",
    structuralReduction: false
  };
}

function sampleLayerRiskImportanceReport() {
  return {
    schemaVersion: 1,
    sourceModelId: "openbmb/MiniCPM5-1B",
    calibration: {
      promptCount: 12,
      contextTokens: 256,
      prefixTokenCounts: [1, 2, 4, 8, 12, 18, 32]
    },
    modules: [
      module("model.layers.1.self_attn.v_proj", "attentionV", 1, 10, 0, 0),
      module("model.layers.0.self_attn.v_proj", "attentionV", 0, 90, 0, 0),
      module("model.layers.1.self_attn.q_proj", "attentionQKO", 1, 10, 0, 0),
      module("model.layers.0.self_attn.q_proj", "attentionQKO", 0, 90, 0, 0)
    ]
  };
}

function sampleLayerRiskPolicy() {
  return {
    schemaVersion: 1,
    policyId: "sample-layer-risk-policy",
    layerCount: 2,
    weights: {
      attentionV: "fp16",
      attentionQKO: "fp16",
      ffn: "fp16",
      embedding: "fp16",
      lmHead: "fp16",
      norms: "fp16"
    },
    layerOverrides: {
      attentionV: {
        1: "int4"
      },
      attentionQKO: {
        1: "int4"
      }
    },
    kvCache: "fp16",
    structuralReduction: false
  };
}

function module(name, component, layerIndex, totalActivationEnergy, topColumnEnergyFraction, topGroupEnergy) {
  return {
    name,
    component,
    layerIndex,
    inputFeatures: 8,
    totalActivationEnergy,
    channelSummary: {
      maxColumnEnergy: totalActivationEnergy * topColumnEnergyFraction,
      topColumnEnergyFraction,
      topColumnsEnergyFraction: topColumnEnergyFraction
    },
    topGroups: [
      {
        groupIndex: 0,
        startColumn: 0,
        endColumnExclusive: 4,
        totalActivationEnergy: topGroupEnergy,
        meanActivationEnergy: topGroupEnergy / 4,
        topColumnIndex: 0,
        topColumnEnergy: totalActivationEnergy * topColumnEnergyFraction
      }
    ]
  };
}
