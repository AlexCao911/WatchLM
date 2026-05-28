import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  REQUIRED_PROMPT_CATEGORIES,
  estimatePromptTokens,
  groupPromptsByCategory,
  loadBenchmarkPrompts,
  validateBenchmarkPrompts
} from "../src/benchmarkPrompts.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(__dirname, "..", "fixtures", "benchmark-prompts.json");

test("required prompt categories cover fidelity and watch utility probes", () => {
  assert.deepEqual(REQUIRED_PROMPT_CATEGORIES, [
    "zh_short_instruction",
    "en_short_instruction",
    "code_small_fix",
    "watch_utility",
    "safety_refusal"
  ]);
});

test("benchmark prompt fixture loads from a path and validates", async () => {
  const prompts = await loadBenchmarkPrompts(fixturePath);
  const result = validateBenchmarkPrompts(prompts);

  assert.equal(result.ok, true);
  assert.deepEqual(result.errors, []);
  assert.equal(prompts.length, 10);
});

test("benchmark prompt fixture loads from a file URL", async () => {
  const prompts = await loadBenchmarkPrompts(pathToFileURL(fixturePath));

  assert.equal(prompts.length, 10);
});

test("each required category has at least two prompts", async () => {
  const prompts = await loadBenchmarkPrompts(fixturePath);
  const grouped = groupPromptsByCategory(prompts);

  for (const category of REQUIRED_PROMPT_CATEGORIES) {
    assert.equal(grouped.get(category).length, 2, category);
  }
});

test("every prompt has the required benchmark fields", async () => {
  const prompts = await loadBenchmarkPrompts(fixturePath);

  for (const prompt of prompts) {
    assert.equal(typeof prompt.id, "string");
    assert.equal(typeof prompt.category, "string");
    assert.equal(typeof prompt.language, "string");
    assert.equal(typeof prompt.input, "string");
    assert.equal(typeof prompt.maxNewTokens, "number");
    assert.equal(Array.isArray(prompt.qualityChecks), true);
    assert.ok(prompt.qualityChecks.length > 0);
  }
});

test("maxNewTokens stays inside the short watch answer envelope", async () => {
  const prompts = await loadBenchmarkPrompts(fixturePath);

  for (const prompt of prompts) {
    assert.ok(prompt.maxNewTokens >= 16, prompt.id);
    assert.ok(prompt.maxNewTokens <= 96, prompt.id);
  }
});

test("prompt text stays compatible with a 256 token smoke baseline", async () => {
  const prompts = await loadBenchmarkPrompts(fixturePath);

  for (const prompt of prompts) {
    assert.ok(estimatePromptTokens(prompt) <= 256, prompt.id);
  }
});

test("validation reports all prompt errors in one result", () => {
  const prompts = [
    {
      id: "",
      category: "unknown",
      language: "",
      input: "",
      maxNewTokens: 8,
      qualityChecks: []
    }
  ];

  const result = validateBenchmarkPrompts(prompts);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.id must be a non-empty string/);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.category is unsupported/);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.language must be a non-empty string/);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.input must be a non-empty string/);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.maxNewTokens must be between 16 and 96/);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.qualityChecks must be a non-empty array/);
  assert.match(result.errors.join("\n"), /missing required category zh_short_instruction/);
});
