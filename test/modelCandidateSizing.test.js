import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  evaluateModelCandidate,
  recommendModelCandidates,
  summarizeCandidateEvaluation,
  validateModelCandidateSuite
} from "../tools/validation/modelCandidateSizing.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(
  __dirname,
  "..",
  "tools",
  "validation",
  "fixtures",
  "model-candidates.json"
);
const fixture = JSON.parse(await readFile(fixturePath, "utf8"));

test("model candidate fixture validates and keeps the 1B baseline as teacher-only", () => {
  const result = validateModelCandidateSuite(fixture);

  assert.equal(result.ok, true, result.errors.join("\n"));
  assert.equal(fixture.candidates.length, 6);

  const baseline = fixture.candidates.find((candidate) => candidate.id === "minicpm5-1b-v-low8");
  const evaluation = evaluateModelCandidate(baseline, "watch-se-2");

  assert.equal(evaluation.gate.ok, false);
  assert.match(evaluation.gate.failures.join("\n"), /artifact .* exceeds .* SE2 planning budget/);
  assert.match(evaluation.gate.failures.join("\n"), /estimated peak RSS .* exceeds .* SE2 planning budget/);
  assert.equal(evaluation.role, "teacher-baseline");
});

test("distilled 350M int4 candidate passes SE2 context-256 planning gate", () => {
  const candidate = fixture.candidates.find((item) => item.id === "distilled-watchlm-350m-int4");
  const evaluation = evaluateModelCandidate(candidate, "watch-se-2");

  assert.equal(evaluation.gate.ok, true, evaluation.gate.failures.join("\n"));
  assert.deepEqual(evaluation.gate.failures, []);
  assert.equal(evaluation.deviceProfile, "watch-se-2");
  assert.equal(evaluation.contextTokens, 256);
  assert.equal(evaluation.estimates.artifactMB, 229);
  assert.equal(evaluation.estimates.peakRSSMB, 477);

  assert.deepEqual(summarizeCandidateEvaluation(evaluation), {
    id: "distilled-watchlm-350m-int4",
    sourceModelId: "watchlm/distilled-350m",
    role: "runtime-candidate",
    modelFamily: "watchlm-distilled",
    conversionPriority: 3,
    conversionRisk: "low",
    deviceProfile: "watch-se-2",
    contextTokens: 256,
    parameterCountMillions: 350,
    weightBits: 4,
    artifactMB: 229,
    peakRSSMB: 477,
    gatePass: true
  });
});

test("candidate gate rejects a 600M int4 candidate that exceeds SE2 RSS budget", () => {
  const candidate = fixture.candidates.find((item) => item.id === "distilled-watchlm-600m-int4");
  const evaluation = evaluateModelCandidate(candidate, "watch-se-2");

  assert.equal(evaluation.gate.ok, false);
  assert.match(evaluation.gate.failures.join("\n"), /estimated peak RSS 927MB exceeds 850MB SE2 planning budget/);
  assert.equal(evaluation.estimates.artifactMB, 385);
});

test("candidate recommendation prioritizes an advanced Qwen3 0.6B text model before riskier stretch models", () => {
  const recommendations = recommendModelCandidates(fixture, "watch-se-2");

  assert.equal(recommendations[0].id, "qwen3-0.6b-int4");
  assert.equal(recommendations[0].sourceModelId, "Qwen/Qwen3-0.6B");
  assert.equal(recommendations[0].recommendation, "convert-next");
  assert.equal(recommendations[0].gatePass, true);
  assert.equal(recommendations[0].conversionRisk, "medium");

  const qwen35 = recommendations.find((item) => item.id === "qwen3.5-0.8b-text-only-int4");
  assert.equal(qwen35.recommendation, "stretch");
  assert.equal(qwen35.gatePass, false);
  assert.equal(qwen35.conversionRisk, "high");
});

test("candidate validation reports schema errors together", () => {
  const result = validateModelCandidateSuite({
    schemaVersion: 1,
    candidates: [
      {
        id: "",
        sourceModelId: "",
        role: "maybe",
        parameterCount: -1,
        contextTokens: 999,
        quantization: { weightBits: 3 },
        architecture: { layers: 0 }
      }
    ]
  });

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /candidates\[0\]\.id must be a non-empty string/);
  assert.match(result.errors.join("\n"), /candidates\[0\]\.role must be teacher-baseline or runtime-candidate/);
  assert.match(result.errors.join("\n"), /candidates\[0\]\.parameterCount must be a positive number/);
  assert.match(result.errors.join("\n"), /candidates\[0\]\.contextTokens must be one of 128, 256, 512/);
  assert.match(result.errors.join("\n"), /candidates\[0\]\.quantization\.weightBits must be 16, 8, or 4/);
  assert.match(result.errors.join("\n"), /candidates\[0\]\.architecture\.kvHeads must be a positive number/);
});
