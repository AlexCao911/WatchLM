import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const cliPath = path.join(repoRoot, "bin", "watchlm-validate.js");
const manifestPath = path.join(repoRoot, "fixtures", "sample-model-manifest.json");
const promptsPath = path.join(repoRoot, "fixtures", "benchmark-prompts.json");
const reportPath = path.join(repoRoot, "fixtures", "sample-benchmark-report.json");

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

test("validates benchmark report fixtures", () => {
  const result = runCli(["report", reportPath]);

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /report ok/);
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
