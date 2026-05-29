import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const python = process.env.PYTHON ?? "python3";
const cliPath = path.join(repoRoot, "tools", "benchmark", "generate-teacher-references.py");
const promptsPath = path.join(repoRoot, "tools", "benchmark", "fixtures", "benchmark-prompts.json");

test("teacher reference CLI can emit a prompt-id keyed sidecar in mock mode", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "watchlm-teacher-"));
  const outputPath = path.join(dir, "teacher-references.json");
  const result = spawnSync(
    python,
    [
      cliPath,
      "--prompts",
      promptsPath,
      "--output",
      outputPath,
      "--prompt-limit",
      "2",
      "--mock-token-ids",
      "10,11,12"
    ],
    {
      cwd: repoRoot,
      encoding: "utf8"
    }
  );

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /wrote teacher references/);

  const sidecar = JSON.parse(readFileSync(outputPath, "utf8"));
  assert.equal(sidecar.schemaVersion, 1);
  assert.equal(sidecar.source, "mock-teacher");
  assert.equal(sidecar.modelId, "openbmb/MiniCPM5-1B");
  assert.equal(sidecar.references.length, 2);
  assert.deepEqual(sidecar.references[0], {
    promptID: "zh-short-001",
    tokenIDs: [10, 11, 12]
  });
});
