import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const contractPath = path.join(repoRoot, "tools", "conversion", "coreml-artifact-contract.json");
const gitignorePath = path.join(repoRoot, ".gitignore");

test("Core ML conversion contract declares source checkpoint and tokenizer identity", async () => {
  const contract = await readContract();

  assert.equal(contract.sourceCheckpoint.id, "openbmb/MiniCPM5-1B");
  assert.ok(contract.sourceCheckpoint.revision || contract.sourceCheckpoint.sha256);
  assert.equal(contract.tokenizer.source, "openbmb/MiniCPM5-1B");
  assert.match(contract.tokenizer.sha256, /^[a-f0-9]{64}$/);
});

test("Core ML conversion contract declares split prefill and decode artifacts", async () => {
  const contract = await readContract();

  assert.match(contract.artifacts.prefillModelPath, /\.mlpackage$/);
  assert.match(contract.artifacts.decodeModelPath, /\.mlpackage$/);
  assert.deepEqual(contract.artifacts.entrypoints, ["prefill", "decode"]);
  assert.ok([256, 512, 1024].includes(contract.contextVariant));
});

test("Core ML conversion contract has separate SE 2 and SE 3 quantized variants", async () => {
  const contract = await readContract();

  assert.deepEqual(Object.keys(contract.artifacts.deviceVariants).sort(), [
    "watch-se-2",
    "watch-se-3"
  ]);
  assert.equal(contract.artifacts.deviceVariants["watch-se-2"].contextVariant, 256);
  assert.equal(contract.artifacts.deviceVariants["watch-se-3"].contextVariant, 512);

  for (const variant of Object.values(contract.artifacts.deviceVariants)) {
    assert.match(variant.prefillModelPath, /\.mlpackage$/);
    assert.match(variant.decodeModelPath, /\.mlpackage$/);
    assert.equal(typeof variant.maxNewTokens, "number");
  }
});

test("Core ML conversion contract declares quantization and logits validation evidence", async () => {
  const contract = await readContract();

  assert.equal(contract.quantizationPolicyId, "mixed-int4-ffn-int8-attn-kv");
  assert.equal(contract.logitsValidation.teacherModelId, "openbmb/MiniCPM5-1B");
  assert.equal(typeof contract.logitsValidation.summary, "string");
  assert.ok(contract.logitsValidation.summary.length > 0);
  assert.equal(typeof contract.logitsValidation.maxAbsoluteError, "number");
});

test("Core ML conversion contract records excluded large artifact paths", async () => {
  const contract = await readContract();

  assert.deepEqual(contract.excludedLargeArtifactPaths, [
    "artifacts/**/*.mlpackage",
    "artifacts/**/*.mlmodelc",
    "artifacts/**/*.gguf",
    "artifacts/**/*.safetensors",
    "artifacts/benchmarks/**/*.json"
  ]);
});

test("gitignore excludes generated model and benchmark artifacts", async () => {
  const gitignore = await readFile(gitignorePath, "utf8");

  for (const pattern of [
    ".build/",
    "artifacts/**/*.mlpackage",
    "artifacts/**/*.mlmodelc",
    "artifacts/**/*.gguf",
    "artifacts/**/*.safetensors",
    "artifacts/benchmarks/**/*.json"
  ]) {
    assert.match(gitignore, new RegExp(escapeRegExp(pattern)));
  }
});

async function readContract() {
  return JSON.parse(await readFile(contractPath, "utf8"));
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
