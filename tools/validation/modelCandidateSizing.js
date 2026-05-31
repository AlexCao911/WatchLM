export const SUPPORTED_CANDIDATE_CONTEXTS = Object.freeze([128, 256, 512]);
export const SUPPORTED_CANDIDATE_ROLES = Object.freeze(["teacher-baseline", "runtime-candidate"]);
export const SUPPORTED_WEIGHT_BITS = Object.freeze([16, 8, 4]);

export const WATCH_MODEL_SIZE_TARGETS = Object.freeze({
  "watch-se-2": Object.freeze({
    label: "SE2",
    maxArtifactBytes: 650_000_000,
    maxPeakRSSBytes: 850_000_000
  }),
  "watch-se-3": Object.freeze({
    label: "SE3",
    maxArtifactBytes: 750_000_000,
    maxPeakRSSBytes: 950_000_000
  })
});

const DEFAULT_KV_BYTES = 2;
const DEFAULT_TOKENIZER_BYTES = 10_000_000;
const DEFAULT_COMPRESSION_OVERHEAD = 1.25;
const DEFAULT_COREML_LOAD_MULTIPLIER = 1.8;
const DEFAULT_TEMPORARY_ARENA_BYTES = 64_000_000;

export function validateModelCandidateSuite(suite) {
  const errors = [];

  if (!isRecord(suite)) {
    return {
      ok: false,
      errors: ["candidate suite must be an object"]
    };
  }

  if (suite.schemaVersion !== 1) {
    errors.push("schemaVersion must be 1");
  }

  if (!Array.isArray(suite.candidates) || suite.candidates.length === 0) {
    errors.push("candidates must be a non-empty array");
  } else {
    suite.candidates.forEach((candidate, index) => {
      validateCandidate(candidate, index, errors);
    });
  }

  return {
    ok: errors.length === 0,
    errors
  };
}

export function assertValidModelCandidateSuite(suite) {
  const result = validateModelCandidateSuite(suite);
  if (!result.ok) {
    throw new Error(`Invalid model candidate suite:\n- ${result.errors.join("\n- ")}`);
  }
}

export function evaluateModelCandidate(candidate, deviceProfile = "watch-se-2") {
  const target = WATCH_MODEL_SIZE_TARGETS[deviceProfile];
  if (!target) {
    throw new Error(`unsupported device profile: ${deviceProfile}`);
  }

  const estimates = estimateModelCandidate(candidate);
  const failures = [];
  if (estimates.artifactBytes > target.maxArtifactBytes) {
    failures.push(
      `artifact ${estimates.artifactMB}MB exceeds ${mb(target.maxArtifactBytes)}MB ${target.label} planning budget`
    );
  }
  if (estimates.peakRSSBytes > target.maxPeakRSSBytes) {
    failures.push(
      `estimated peak RSS ${estimates.peakRSSMB}MB exceeds ${mb(target.maxPeakRSSBytes)}MB ${target.label} planning budget`
    );
  }

  return {
    id: candidate.id,
    sourceModelId: candidate.sourceModelId,
    role: candidate.role,
    deviceProfile,
    contextTokens: candidate.contextTokens,
    architecture: candidate.architecture,
    quantization: candidate.quantization,
    estimates,
    gate: {
      ok: failures.length === 0,
      failures,
      targets: {
        maxArtifactMB: mb(target.maxArtifactBytes),
        maxPeakRSSMB: mb(target.maxPeakRSSBytes)
      }
    }
  };
}

export function summarizeCandidateEvaluation(evaluation) {
  return {
    id: evaluation.id,
    role: evaluation.role,
    deviceProfile: evaluation.deviceProfile,
    contextTokens: evaluation.contextTokens,
    parameterCountMillions: Math.round(evaluation.architecture.parameterCount / 1_000_000),
    weightBits: evaluation.quantization.weightBits,
    artifactMB: evaluation.estimates.artifactMB,
    peakRSSMB: evaluation.estimates.peakRSSMB,
    gatePass: evaluation.gate.ok
  };
}

function estimateModelCandidate(candidate) {
  if (isPositiveNumber(candidate.measuredArtifactBytes) && isPositiveNumber(candidate.measuredPeakRSSBytes)) {
    return {
      weightBytes: undefined,
      kvBytes: estimateKVBytes(candidate),
      artifactBytes: candidate.measuredArtifactBytes,
      artifactMB: mb(candidate.measuredArtifactBytes),
      peakRSSBytes: candidate.measuredPeakRSSBytes,
      peakRSSMB: mb(candidate.measuredPeakRSSBytes),
      source: "measured"
    };
  }

  const parameterCount = candidate.architecture.parameterCount;
  const weightBits = candidate.quantization.weightBits;
  const compressionOverhead = candidate.quantization.compressionOverhead ?? DEFAULT_COMPRESSION_OVERHEAD;
  const tokenizerBytes = candidate.tokenizerBytes ?? DEFAULT_TOKENIZER_BYTES;
  const coreMLLoadMultiplier = candidate.runtime?.coreMLLoadMultiplier ?? DEFAULT_COREML_LOAD_MULTIPLIER;
  const temporaryArenaBytes = candidate.runtime?.temporaryArenaBytes ?? DEFAULT_TEMPORARY_ARENA_BYTES;

  const weightBytes = parameterCount * (weightBits / 8) * compressionOverhead;
  const kvBytes = estimateKVBytes(candidate);
  const artifactBytes = weightBytes + tokenizerBytes;
  const peakRSSBytes = artifactBytes * coreMLLoadMultiplier + kvBytes + temporaryArenaBytes;

  return {
    weightBytes: Math.round(weightBytes),
    kvBytes: Math.round(kvBytes),
    artifactBytes: Math.round(artifactBytes),
    artifactMB: mb(artifactBytes),
    peakRSSBytes: Math.round(peakRSSBytes),
    peakRSSMB: mb(peakRSSBytes),
    source: "estimated"
  };
}

function estimateKVBytes(candidate) {
  const architecture = candidate.architecture ?? {};
  const layers = architecture.layers;
  const kvHeads = architecture.kvHeads;
  const headDimension = architecture.headDimension;
  const contextTokens = candidate.contextTokens;
  const kvBytes = candidate.quantization?.kvBytes ?? DEFAULT_KV_BYTES;
  if (![layers, kvHeads, headDimension, contextTokens, kvBytes].every(isPositiveNumber)) {
    return 0;
  }
  return layers * 2 * kvHeads * contextTokens * headDimension * kvBytes;
}

function validateCandidate(candidate, index, errors) {
  const prefix = `candidates[${index}]`;
  if (!isRecord(candidate)) {
    errors.push(`${prefix} must be an object`);
    return;
  }

  if (!isNonEmptyString(candidate.id)) {
    errors.push(`${prefix}.id must be a non-empty string`);
  }
  if (!isNonEmptyString(candidate.sourceModelId)) {
    errors.push(`${prefix}.sourceModelId must be a non-empty string`);
  }
  if (!SUPPORTED_CANDIDATE_ROLES.includes(candidate.role)) {
    errors.push(`${prefix}.role must be teacher-baseline or runtime-candidate`);
  }
  if (!isPositiveNumber(candidate.parameterCount)) {
    errors.push(`${prefix}.parameterCount must be a positive number`);
  }
  if (!SUPPORTED_CANDIDATE_CONTEXTS.includes(candidate.contextTokens)) {
    errors.push(`${prefix}.contextTokens must be one of ${SUPPORTED_CANDIDATE_CONTEXTS.join(", ")}`);
  }

  if (!isRecord(candidate.quantization)) {
    errors.push(`${prefix}.quantization must be an object`);
  } else if (!SUPPORTED_WEIGHT_BITS.includes(candidate.quantization.weightBits)) {
    errors.push(`${prefix}.quantization.weightBits must be 16, 8, or 4`);
  }

  if (!isRecord(candidate.architecture)) {
    errors.push(`${prefix}.architecture must be an object`);
    return;
  }

  if (!isPositiveNumber(candidate.architecture.layers)) {
    errors.push(`${prefix}.architecture.layers must be a positive number`);
  }
  if (!isPositiveNumber(candidate.architecture.kvHeads)) {
    errors.push(`${prefix}.architecture.kvHeads must be a positive number`);
  }
  if (!isPositiveNumber(candidate.architecture.headDimension)) {
    errors.push(`${prefix}.architecture.headDimension must be a positive number`);
  }
  if (candidate.architecture.parameterCount !== candidate.parameterCount) {
    errors.push(`${prefix}.architecture.parameterCount must match parameterCount`);
  }
}

function mb(bytes) {
  return Math.round(bytes / 1_000_000);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim() !== "";
}

function isPositiveNumber(value) {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}
