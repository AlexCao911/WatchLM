import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const cliPath = path.join(repoRoot, "tools", "validation", "watchlm-validate.js");
const manifestPath = path.join(
  repoRoot,
  "tools",
  "validation",
  "fixtures",
  "sample-model-manifest.json"
);
const promptsPath = path.join(
  repoRoot,
  "tools",
  "benchmark",
  "fixtures",
  "benchmark-prompts.json"
);
const calibrationPromptsPath = path.join(
  repoRoot,
  "tools",
  "benchmark",
  "fixtures",
  "calibration-prompts.json"
);
const reportPath = path.join(
  repoRoot,
  "tools",
  "benchmark",
  "fixtures",
  "sample-benchmark-report.json"
);
const candidatesPath = path.join(
  repoRoot,
  "tools",
  "validation",
  "fixtures",
  "model-candidates.json"
);

function runCli(args) {
  return spawnSync(process.execPath, [cliPath, ...args], {
    cwd: repoRoot,
    encoding: "utf8"
  });
}

test("validates a model manifest fixture", () => {
  const result = runCli(["manifest", manifestPath]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /manifest ok/);
});

test("validates benchmark prompt fixtures", () => {
  const result = runCli(["prompts", promptsPath]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /prompts ok/);
});

test("validates calibration prompt fixtures", () => {
  const result = runCli(["calibration-prompts", calibrationPromptsPath]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /calibration prompts ok: 12 prompts, prefixes=1,2,4,8,12,18,32/);
});

test("validates benchmark report fixtures", () => {
  const result = runCli(["report", reportPath]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /report ok/);
});

test("validates model candidate sizing fixtures", () => {
  const result = runCli(["candidates", candidatesPath]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /candidates ok: 6 candidates, 3 passing SE2 gate/);
  assert.match(result.stdout, /recommended next: qwen3-0.6b-int4 \(Qwen\/Qwen3-0.6B\)/);
  assert.match(result.stdout, /distilled-watchlm-350m-int4: pass/);
  assert.match(result.stdout, /qwen3-0.6b-int4: pass/);
  assert.match(result.stdout, /qwen3.5-0.8b-text-only-int4: fail/);
  assert.match(result.stdout, /minicpm5-1b-v-low8: fail/);
});

test("validates all host evidence contracts in one command", () => {
  const result = runCli([
    "all",
    "--manifest",
    manifestPath,
    "--prompts",
    promptsPath,
    "--report",
    reportPath
  ]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /manifest ok/);
  assert.match(result.stdout, /prompts ok/);
  assert.match(result.stdout, /report ok/);
});

test("invalid manifest exits non-zero and prints validation errors", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "watchlm-cli-"));
  const invalidPath = path.join(dir, "invalid-manifest.json");
  writeFileSync(
    invalidPath,
    JSON.stringify({
      model: { id: "openbmb/MiniCPM5-1B" },
      runtime: { type: "llama.cpp" }
    })
  );

  const result = runCli(["manifest", invalidPath]);

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /runtime\.type must be coreml-mlprogram/);
});
