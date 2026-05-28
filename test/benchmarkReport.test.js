import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  DEVICE_TARGETS,
  evaluateBenchmarkGates,
  requiresFallbackEvidence,
  summarizeBenchmarkReport,
  validateBenchmarkReport
} from "../src/benchmarkReport.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(__dirname, "..", "fixtures", "sample-benchmark-report.json");
const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
const [se2Report, se3Report] = fixture.reports;

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

test("device targets encode SE 2 and SE 3 usability gates", () => {
  assert.deepEqual(DEVICE_TARGETS, {
    "watch-se-2": {
      maxFirstTokenMs: 5000,
      minDecodeTokensPerSecond: 1.5
    },
    "watch-se-3": {
      maxFirstTokenMs: 3000,
      minDecodeTokensPerSecond: 3
    }
  });
});

test("sample reports include all required benchmark evidence", () => {
  for (const report of fixture.reports) {
    const result = validateBenchmarkReport(report);

    assert.equal(result.ok, true, result.errors.join("\n"));
    assert.deepEqual(result.errors, []);
  }
});

test("report validation requires model, device, runtime, timing, memory, thermal, quality, and fallback fields", () => {
  const result = validateBenchmarkReport({});

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /sourceModelId must be openbmb\/MiniCPM5-1B/);
  assert.match(result.errors.join("\n"), /deviceProfile must be watch-se-2 or watch-se-3/);
  assert.match(result.errors.join("\n"), /runtime must be coreml-mlprogram/);
  assert.match(result.errors.join("\n"), /contextVariant must be one of 256, 512, 1024/);
  assert.match(result.errors.join("\n"), /artifact\.sizeBytes must be a positive number/);
  assert.match(result.errors.join("\n"), /timings\.loadMs must be a non-negative number/);
  assert.match(result.errors.join("\n"), /timings\.prefillMs must be a non-negative number/);
  assert.match(result.errors.join("\n"), /timings\.firstTokenMs must be a non-negative number/);
  assert.match(result.errors.join("\n"), /timings\.decodeTokensPerSecond must be a positive number/);
  assert.match(result.errors.join("\n"), /memory\.peakResidentMB must be a positive number/);
  assert.match(result.errors.join("\n"), /thermal\.fiveTurnStates must include five short-turn states/);
  assert.match(result.errors.join("\n"), /qualityDrift\.summary must be a non-empty string/);
  assert.match(result.errors.join("\n"), /fallbackDecision\.status must be present/);
});

test("SE 3 report passes when first token and decode speed meet targets", () => {
  const result = evaluateBenchmarkGates(se3Report);

  assert.equal(result.ok, true);
  assert.deepEqual(result.failures, []);
  assert.equal(result.targets.maxFirstTokenMs, 3000);
  assert.equal(result.targets.minDecodeTokensPerSecond, 3);
});

test("SE 2 report passes its softer watch target", () => {
  const result = evaluateBenchmarkGates(se2Report);

  assert.equal(result.ok, true);
  assert.deepEqual(result.failures, []);
  assert.equal(result.targets.maxFirstTokenMs, 5000);
  assert.equal(result.targets.minDecodeTokensPerSecond, 1.5);
});

test("gate evaluation reports first-token and decode failures together", () => {
  const report = clone(se3Report);
  report.timings.firstTokenMs = 3100;
  report.timings.decodeTokensPerSecond = 2.9;

  const result = evaluateBenchmarkGates(report);

  assert.equal(result.ok, false);
  assert.match(result.failures.join("\n"), /first token 3100ms exceeds 3000ms target/);
  assert.match(result.failures.join("\n"), /decode 2.9 tok\/s is below 3 tok\/s target/);
});

test("fallback decisions require explicit evidence", () => {
  const report = clone(se3Report);
  report.fallbackDecision = {
    status: "required",
    action: "reduce-context",
    reason: "Decode speed is below target",
    evidence: []
  };

  const result = validateBenchmarkReport(report);

  assert.equal(requiresFallbackEvidence(report), true);
  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /fallbackDecision\.evidence must include at least one report section/);
});

test("report summary exposes audit-friendly benchmark details", () => {
  const summary = summarizeBenchmarkReport(se3Report);

  assert.deepEqual(summary, {
    id: "sample-se3-target",
    sourceModelId: "openbmb/MiniCPM5-1B",
    deviceProfile: "watch-se-3",
    runtime: "coreml-mlprogram",
    contextVariant: 512,
    artifactSizeMB: 620,
    firstTokenMs: 2800,
    decodeTokensPerSecond: 3.2,
    peakResidentMB: 780,
    gatesPass: true,
    fallbackStatus: "not-required"
  });
});
