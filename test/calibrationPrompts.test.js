import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  REQUIRED_CALIBRATION_CATEGORIES,
  loadCalibrationPromptSuite,
  toBenchmarkPrompts,
  validateCalibrationPromptSuite
} from "../tools/benchmark/calibrationPrompts.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(
  __dirname,
  "..",
  "tools",
  "benchmark",
  "fixtures",
  "calibration-prompts.json"
);

test("calibration prompt fixture covers WatchLM quantization categories", async () => {
  const suite = await loadCalibrationPromptSuite(fixturePath);
  const result = validateCalibrationPromptSuite(suite);

  assert.equal(result.ok, true, result.errors.join("\n"));
  assert.deepEqual(suite.prefixTokenCounts, [1, 2, 4, 8, 12, 18, 32]);
  assert.equal(suite.contextTokens, 256);
  assert.equal(suite.promptFormat, "minicpm5-chat-template-no-think");
  assert.equal(suite.prompts.length, 12);

  const categories = new Set(suite.prompts.map((prompt) => prompt.category));
  assert.deepEqual(categories, new Set(REQUIRED_CALIBRATION_CATEGORIES));
  assert.ok(
    suite.prompts.every((prompt) => prompt.renderedPrompt.includes("<think>\n\n</think>\n\n"))
  );
});

test("calibration prompt suite converts to benchmark prompts for diagnostics", async () => {
  const suite = await loadCalibrationPromptSuite(fixturePath);
  const prompts = toBenchmarkPrompts(suite, { maxNewTokens: 2 });

  assert.equal(prompts.length, suite.prompts.length);
  assert.equal(prompts[0].id, "cal-zh-short-001");
  assert.equal(prompts[0].input, suite.prompts[0].renderedPrompt);
  assert.equal(prompts[0].maxNewTokens, 2);
  assert.deepEqual(prompts[0].qualityChecks, suite.prompts[0].tags);
});

test("calibration prompt validation reports all structural errors", () => {
  const result = validateCalibrationPromptSuite({
    schemaVersion: 2,
    modelId: "other/model",
    tokenizerSource: "",
    contextTokens: 128,
    promptFormat: "raw",
    prefixTokenCounts: [4, 2, 512],
    prompts: [
      {
        id: "",
        category: "unknown",
        language: "",
        messages: [],
        renderedPrompt: "hello",
        maxNewTokens: 0,
        tags: []
      }
    ]
  });

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /schemaVersion must be 1/);
  assert.match(result.errors.join("\n"), /modelId must be openbmb\/MiniCPM5-1B/);
  assert.match(result.errors.join("\n"), /prefixTokenCounts must be strictly increasing/);
  assert.match(result.errors.join("\n"), /prompt\[0\]\.renderedPrompt must use the MiniCPM no-think assistant prefix/);
  assert.match(result.errors.join("\n"), /missing required category zh_short_instruction/);
});
