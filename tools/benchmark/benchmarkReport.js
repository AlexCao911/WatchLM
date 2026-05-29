import {
  EXPECTED_MODEL_ID,
  EXPECTED_RUNTIME,
  SUPPORTED_CONTEXT_VARIANTS,
  SUPPORTED_DEVICE_PROFILES
} from "../validation/modelManifest.js";

export const DEVICE_TARGETS = Object.freeze({
  "watch-se-2": Object.freeze({
    maxFirstTokenMs: 5000,
    minDecodeTokensPerSecond: 1.5
  }),
  "watch-se-3": Object.freeze({
    maxFirstTokenMs: 3000,
    minDecodeTokensPerSecond: 3
  })
});

const FALLBACK_EVIDENCE_STATUSES = new Set(["required", "proposed", "applied"]);

export function validateBenchmarkReport(report) {
  const errors = [];

  if (!isRecord(report)) {
    return {
      ok: false,
      errors: ["report must be an object"]
    };
  }

  if (report.sourceModelId !== EXPECTED_MODEL_ID) {
    errors.push(`sourceModelId must be ${EXPECTED_MODEL_ID}`);
  }

  if (!SUPPORTED_DEVICE_PROFILES.includes(report.deviceProfile)) {
    errors.push("deviceProfile must be watch-se-2 or watch-se-3");
  }

  if (report.runtime !== EXPECTED_RUNTIME) {
    errors.push(`runtime must be ${EXPECTED_RUNTIME}`);
  }

  if (!SUPPORTED_CONTEXT_VARIANTS.includes(report.contextVariant)) {
    errors.push(`contextVariant must be one of ${SUPPORTED_CONTEXT_VARIANTS.join(", ")}`);
  }

  validateArtifact(report, errors);
  validateTimings(report, errors);
  validateMemory(report, errors);
  validateThermal(report, errors);
  validateQualityDrift(report, errors);
  validateFallbackDecision(report, errors);

  return {
    ok: errors.length === 0,
    errors
  };
}

export function evaluateBenchmarkGates(report) {
  const validation = validateBenchmarkReport(report);
  const targets = DEVICE_TARGETS[report?.deviceProfile];
  const failures = [...validation.errors];

  if (targets) {
    if (report.timings.firstTokenMs > targets.maxFirstTokenMs) {
      failures.push(
        `first token ${report.timings.firstTokenMs}ms exceeds ${targets.maxFirstTokenMs}ms target`
      );
    }

    if (report.timings.decodeTokensPerSecond < targets.minDecodeTokensPerSecond) {
      failures.push(
        `decode ${report.timings.decodeTokensPerSecond} tok/s is below ${targets.minDecodeTokensPerSecond} tok/s target`
      );
    }
  }

  return {
    ok: failures.length === 0,
    failures,
    targets,
    metrics: {
      firstTokenMs: report?.timings?.firstTokenMs,
      decodeTokensPerSecond: report?.timings?.decodeTokensPerSecond
    }
  };
}

export function summarizeBenchmarkReport(report) {
  const gates = evaluateBenchmarkGates(report);

  return {
    id: report.id,
    sourceModelId: report.sourceModelId,
    deviceProfile: report.deviceProfile,
    runtime: report.runtime,
    contextVariant: report.contextVariant,
    artifactSizeMB: Math.round(report.artifact.sizeBytes / 1_000_000),
    firstTokenMs: report.timings.firstTokenMs,
    decodeTokensPerSecond: report.timings.decodeTokensPerSecond,
    peakResidentMB: report.memory.peakResidentMB,
    gatesPass: gates.ok,
    fallbackStatus: report.fallbackDecision.status
  };
}

export function requiresFallbackEvidence(report) {
  return FALLBACK_EVIDENCE_STATUSES.has(report?.fallbackDecision?.status);
}

function validateArtifact(report, errors) {
  if (!isPositiveNumber(report.artifact?.sizeBytes)) {
    errors.push("artifact.sizeBytes must be a positive number");
  }
}

function validateTimings(report, errors) {
  if (!isNonNegativeNumber(report.timings?.loadMs)) {
    errors.push("timings.loadMs must be a non-negative number");
  }

  if (!isNonNegativeNumber(report.timings?.prefillMs)) {
    errors.push("timings.prefillMs must be a non-negative number");
  }

  if (!isNonNegativeNumber(report.timings?.firstTokenMs)) {
    errors.push("timings.firstTokenMs must be a non-negative number");
  }

  if (!isPositiveNumber(report.timings?.decodeTokensPerSecond)) {
    errors.push("timings.decodeTokensPerSecond must be a positive number");
  }
}

function validateMemory(report, errors) {
  if (!isPositiveNumber(report.memory?.peakResidentMB)) {
    errors.push("memory.peakResidentMB must be a positive number");
  }
}

function validateThermal(report, errors) {
  if (!Array.isArray(report.thermal?.fiveTurnStates) || report.thermal.fiveTurnStates.length !== 5) {
    errors.push("thermal.fiveTurnStates must include five short-turn states");
  }
}

function validateQualityDrift(report, errors) {
  if (typeof report.qualityDrift?.summary !== "string" || report.qualityDrift.summary.trim() === "") {
    errors.push("qualityDrift.summary must be a non-empty string");
  }
}

function validateFallbackDecision(report, errors) {
  if (typeof report.fallbackDecision?.status !== "string" || report.fallbackDecision.status.trim() === "") {
    errors.push("fallbackDecision.status must be present");
    return;
  }

  if (
    requiresFallbackEvidence(report) &&
    (!Array.isArray(report.fallbackDecision.evidence) || report.fallbackDecision.evidence.length === 0)
  ) {
    errors.push("fallbackDecision.evidence must include at least one report section");
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isPositiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function isNonNegativeNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}
