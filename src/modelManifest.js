export const SUPPORTED_CONTEXT_VARIANTS = Object.freeze([256, 512, 1024]);
export const SUPPORTED_DEVICE_PROFILES = Object.freeze(["watch-se-2", "watch-se-3"]);
export const EXPECTED_MODEL_ID = "openbmb/MiniCPM5-1B";
export const EXPECTED_RUNTIME = "coreml-mlprogram";
export const EXPECTED_ARCHITECTURE = Object.freeze({
  layers: 24,
  hiddenSize: 1536,
  queryHeads: 16,
  kvHeads: 2,
  tokenizerSource: EXPECTED_MODEL_ID
});

export function validateModelManifest(manifest) {
  const errors = [];
  const warnings = [];

  if (!isRecord(manifest)) {
    return {
      ok: false,
      errors: ["manifest must be an object"],
      warnings
    };
  }

  validateModel(manifest, errors);
  validateRuntime(manifest, errors);
  validateArchitecture(manifest, errors);
  validateDeviceProfiles(manifest, errors);
  validateContextVariants(manifest, errors);
  validateAsset(manifest, errors);
  validateQuantization(manifest, errors);
  validateFallbackPolicy(manifest, warnings);

  return {
    ok: errors.length === 0,
    errors,
    warnings
  };
}

export function assertValidModelManifest(manifest) {
  const result = validateModelManifest(manifest);
  if (!result.ok) {
    throw new Error(`Invalid model manifest:\n- ${result.errors.join("\n- ")}`);
  }
}

export function selectContextVariant(manifest, deviceProfile, requestedTokens) {
  assertValidModelManifest(manifest);

  const profile = manifest.deviceProfiles[deviceProfile];
  if (!profile) {
    throw new Error(`Unsupported device profile: ${deviceProfile}`);
  }

  const variants = [...manifest.contextVariants].sort((left, right) => left - right);
  const requested = Number.isFinite(requestedTokens)
    ? requestedTokens
    : profile.defaultContextVariant;
  const selected = variants.filter((variant) => variant <= requested).at(-1);

  return selected ?? variants[0];
}

export function selectModelArtifact(manifest, deviceProfile, requestedTokens) {
  const contextVariant = selectContextVariant(manifest, deviceProfile, requestedTokens);
  const variant = manifest.asset?.variants?.[String(contextVariant)];

  if (variant) {
    return {
      contextVariant,
      deviceProfile: variant.deviceProfile,
      prefillPath: variant.prefillPath,
      decodePath: variant.decodePath,
      sha256: variant.sha256
    };
  }

  return {
    contextVariant,
    deviceProfile,
    prefillPath: manifest.asset.prefillPath,
    decodePath: manifest.asset.decodePath,
    sha256: manifest.asset.sha256
  };
}

export function summarizeModelManifest(manifest) {
  assertValidModelManifest(manifest);

  return {
    modelId: manifest.model.id,
    runtime: manifest.runtime.type,
    deviceProfiles: SUPPORTED_DEVICE_PROFILES.filter(
      (profile) => profile in manifest.deviceProfiles
    ),
    contextVariants: [...manifest.contextVariants],
    assetStorage: manifest.asset.storage,
    assetVariants: Object.keys(manifest.asset.variants ?? {}).map(Number).sort((left, right) => left - right),
    quantizationStrategy: manifest.quantization.strategy,
    kvCachePrecision: manifest.quantization.kvCache
  };
}

function validateModel(manifest, errors) {
  if (manifest.model?.id !== EXPECTED_MODEL_ID) {
    errors.push(`model.id must be ${EXPECTED_MODEL_ID}`);
  }
}

function validateRuntime(manifest, errors) {
  if (manifest.runtime?.type !== EXPECTED_RUNTIME) {
    errors.push(`runtime.type must be ${EXPECTED_RUNTIME}`);
  }

  const entrypoints = manifest.runtime?.entrypoints;
  if (!Array.isArray(entrypoints) || !entrypoints.includes("prefill") || !entrypoints.includes("decode")) {
    errors.push("runtime.entrypoints must include prefill and decode");
  }
}

function validateArchitecture(manifest, errors) {
  const architecture = manifest.architecture ?? {};
  if (architecture.layers !== EXPECTED_ARCHITECTURE.layers) {
    errors.push(`architecture.layers must be ${EXPECTED_ARCHITECTURE.layers}`);
  }

  if (architecture.hiddenSize !== EXPECTED_ARCHITECTURE.hiddenSize) {
    errors.push(`architecture.hiddenSize must be ${EXPECTED_ARCHITECTURE.hiddenSize}`);
  }

  if (architecture.queryHeads !== EXPECTED_ARCHITECTURE.queryHeads) {
    errors.push(`architecture.queryHeads must be ${EXPECTED_ARCHITECTURE.queryHeads}`);
  }

  if (architecture.kvHeads !== EXPECTED_ARCHITECTURE.kvHeads) {
    errors.push(`architecture.kvHeads must be ${EXPECTED_ARCHITECTURE.kvHeads}`);
  }

  if (architecture.tokenizer?.source !== EXPECTED_ARCHITECTURE.tokenizerSource) {
    errors.push(`architecture.tokenizer.source must be ${EXPECTED_ARCHITECTURE.tokenizerSource}`);
  }

  if (architecture.tokenizer?.preserved !== true) {
    errors.push("tokenizer must be preserved");
  }

  if (architecture.tokenizer?.vocabularyPreserved !== true) {
    errors.push("vocabulary must be preserved");
  }
}

function validateDeviceProfiles(manifest, errors) {
  if (!isRecord(manifest.deviceProfiles)) {
    errors.push("deviceProfiles must be an object");
    return;
  }

  for (const profile of SUPPORTED_DEVICE_PROFILES) {
    const config = manifest.deviceProfiles[profile];
    if (!isRecord(config)) {
      errors.push(`deviceProfiles.${profile} must be present`);
      continue;
    }

    if (!SUPPORTED_CONTEXT_VARIANTS.includes(config.defaultContextVariant)) {
      errors.push(`deviceProfiles.${profile}.defaultContextVariant must be supported`);
    }
  }
}

function validateContextVariants(manifest, errors) {
  if (!Array.isArray(manifest.contextVariants) || manifest.contextVariants.length === 0) {
    errors.push("contextVariants must be a non-empty array");
    return;
  }

  for (const variant of manifest.contextVariants) {
    if (!SUPPORTED_CONTEXT_VARIANTS.includes(variant)) {
      errors.push(`unsupported context variant ${variant}`);
    }
  }
}

function validateAsset(manifest, errors) {
  if (manifest.asset?.storage === "app-bundle") {
    errors.push("asset.storage must not be app-bundle");
  }

  if (manifest.asset?.storage !== "application-support") {
    errors.push("asset.storage must be application-support");
  }

  if (typeof manifest.asset?.sha256 !== "string" || manifest.asset.sha256.length !== 64) {
    errors.push("asset.sha256 must be a 64-character hex digest");
  }

  validateAssetVariants(manifest, errors);
}

function validateAssetVariants(manifest, errors) {
  const variants = manifest.asset?.variants;
  if (variants === undefined) {
    return;
  }

  if (!isRecord(variants)) {
    errors.push("asset.variants must be an object when present");
    return;
  }

  for (const profile of SUPPORTED_DEVICE_PROFILES) {
    const contextVariant = manifest.deviceProfiles?.[profile]?.defaultContextVariant;
    const variant = variants[String(contextVariant)];
    if (!isRecord(variant)) {
      errors.push(`asset.variants.${contextVariant} must be present for ${profile}`);
      continue;
    }

    if (variant.deviceProfile !== profile) {
      errors.push(`asset.variants.${contextVariant}.deviceProfile must be ${profile}`);
    }

    if (typeof variant.prefillPath !== "string" || !variant.prefillPath.endsWith(".mlpackage")) {
      errors.push(`asset.variants.${contextVariant}.prefillPath must be an mlpackage path`);
    }

    if (typeof variant.decodePath !== "string" || !variant.decodePath.endsWith(".mlpackage")) {
      errors.push(`asset.variants.${contextVariant}.decodePath must be an mlpackage path`);
    }

    if (typeof variant.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(variant.sha256)) {
      errors.push(`asset.variants.${contextVariant}.sha256 must be a 64-character hex digest`);
    }
  }
}

function validateQuantization(manifest, errors) {
  const quantization = manifest.quantization ?? {};
  if (quantization.strategy !== "mixed-precision-fidelity-first") {
    errors.push("quantization.strategy must be mixed-precision-fidelity-first");
  }

  if (quantization.kvCache !== "int8") {
    errors.push("quantization.kvCache must be int8");
  }

  if (quantization.structuralReduction !== false) {
    errors.push("structuralReduction must be false");
  }

  if (!isRecord(quantization.weights)) {
    errors.push("quantization.weights must describe per-component precision");
    return;
  }

  for (const component of ["embedding", "lmHead", "norms", "attentionQKO", "ffn"]) {
    if (typeof quantization.weights[component] !== "string") {
      errors.push(`quantization.weights.${component} must be present`);
    }
  }
}

function validateFallbackPolicy(manifest, warnings) {
  if (manifest.fallbackPolicy?.requiresBenchmarkEvidence !== true) {
    warnings.push("fallbackPolicy.requiresBenchmarkEvidence should be true");
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
