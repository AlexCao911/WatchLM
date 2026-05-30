import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, realpath, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const python = path.join(repoRoot, ".venv", "bin", "python");
const suggestionScript = path.join(repoRoot, "tools", "conversion", "suggest-importance-policy.py");
const conversionScript = path.join(repoRoot, "tools", "conversion", "convert-minicpm5-coreml.py");

test("importance policy suggestion chooses low-energy layers and preserves protected edges", async () => {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "watchlm-policy-suggestion-"));
  const reportPath = path.join(tempDir, "importance.json");
  const policyPath = path.join(tempDir, "policy.json");
  await writeFile(reportPath, JSON.stringify(sampleImportanceReport()), "utf8");

  const { stdout } = await execFileAsync(python, [
    suggestionScript,
    "--importance-report",
    reportPath,
    "--component",
    "attentionV",
    "--candidate-count",
    "3",
    "--protected-edge-layer-count",
    "2",
    "--exclude-layers",
    "4",
    "--policy-id",
    "importance-attention-v-low3",
    "--output",
    policyPath
  ], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });
  const policy = JSON.parse(stdout);
  const written = JSON.parse(await readFile(policyPath, "utf8"));

  assert.deepEqual(written, policy);
  assert.equal(policy.policyId, "importance-attention-v-low3");
  assert.deepEqual(policy.weights, {
    embedding: "fp16",
    lmHead: "fp16",
    norms: "fp16",
    attentionQKO: "fp16",
    attentionV: "fp16",
    ffn: "fp16"
  });
  assert.deepEqual(policy.layerOverrides, {
    attentionV: {
      3: "int4",
      2: "int4",
      5: "int4"
    }
  });
  assert.equal(policy.candidateEvidence.sourceReport, await realpath(reportPath));
  assert.deepEqual(policy.candidateEvidence.selectedLayers.map((item) => item.layerIndex), [3, 2, 5]);
  assert.deepEqual(policy.candidateEvidence.excludedLayers, [0, 1, 4, 6, 7]);

  const { stdout: described } = await execFileAsync(python, [
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
  const plan = JSON.parse(described);
  assert.equal(plan.layerPrecision["3"].attentionV, "int4");
  assert.equal(plan.layerPrecision["2"].attentionV, "int4");
  assert.equal(plan.layerPrecision["5"].attentionV, "int4");
  assert.equal(plan.layerPrecision["4"].attentionV, "fp16");
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
    layerSummary: [
      layer(0, { attentionV: 1 }),
      layer(1, { attentionV: 2 }),
      layer(2, { attentionV: 20 }),
      layer(3, { attentionV: 10 }),
      layer(4, { attentionV: 0.5 }),
      layer(5, { attentionV: 30 }),
      layer(6, { attentionV: 3 }),
      layer(7, { attentionV: 4 })
    ]
  };
}

function layer(layerIndex, componentTotals) {
  return {
    layerIndex,
    moduleCount: 1,
    totalActivationEnergy: Object.values(componentTotals).reduce((total, value) => total + value, 0),
    componentTotals
  };
}
