export const SUPPORTED_CONTEXT_VARIANTS = Object.freeze([256, 512, 1024]);
export const SUPPORTED_DEVICE_PROFILES = Object.freeze(["watch-se-2", "watch-se-3"]);
export const SUPPORTED_KV_CACHE_MODES = Object.freeze([
  "stateful-preferred",
  "slot-ring",
  "contiguous-sliding"
]);
export const SUPPORTED_GRAPH_INTERFACES = Object.freeze([
  "logits-layered-kv",
  "stateful-kv",
  "stateful-step-kv"
]);
export const STATEFUL_GRAPH_INTERFACES = Object.freeze([
  "stateful-kv",
  "stateful-step-kv"
]);
export const EXPECTED_MODEL_ID = "openbmb/MiniCPM5-1B";
export const EXPECTED_RUNTIME = "coreml-mlprogram";
export const EXPECTED_GRAPH_SCHEMA = Object.freeze({
  interface: "logits-layered-kv",
  layerCount: 24,
  kvHeads: 2,
  headDimension: 128,
  prefill: Object.freeze({
    inputIDs: "input_ids",
    positionIDs: "position_ids",
    causalMask: "causal_mask",
    logits: "logits",
    keyPrefix: "present_key_",
    valuePrefix: "present_value_"
  }),
  decode: Object.freeze({
    tokenID: "token_id",
    positionID: "position_id",
    causalMask: "causal_mask",
    logits: "logits",
    pastKeyPrefix: "past_key_",
    pastValuePrefix: "past_value_",
    newKeyPrefix: "new_key_",
    newValuePrefix: "new_value_"
  })
});
export const EXPECTED_ARCHITECTURE = Object.freeze({
  layers: 24,
  hiddenSize: 1536,
  queryHeads: 16,
  kvHeads: 2,
  tokenizerSource: EXPECTED_MODEL_ID
});
const SUPPORTED_PRECISIONS = Object.freeze(["fp16", "int8", "int4"]);
const QUANTIZED_WEIGHT_COMPONENTS = Object.freeze([
  "embedding",
  "lmHead",
  "norms",
  "attentionQKO",
  "attentionV",
  "ffn"
]);
const TRANSFORMER_WEIGHT_COMPONENTS = Object.freeze(["attentionQKO", "attentionV", "ffn"]);

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
      tokenizerPath: variant.tokenizerPath ?? manifest.asset.tokenizerPath,
      sha256: variant.sha256,
      prefillSHA256: variant.prefillSHA256,
      decodeSHA256: variant.decodeSHA256,
      tokenizerSHA256: variant.tokenizerSHA256 ?? manifest.asset.tokenizerSHA256
    };
  }

  return {
    contextVariant,
    deviceProfile,
    prefillPath: manifest.asset.prefillPath,
    decodePath: manifest.asset.decodePath,
    tokenizerPath: manifest.asset.tokenizerPath,
    sha256: manifest.asset.sha256,
    prefillSHA256: manifest.asset.prefillSHA256,
    decodeSHA256: manifest.asset.decodeSHA256,
    tokenizerSHA256: manifest.asset.tokenizerSHA256
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
    kvCacheMode: manifest.runtime.kvCacheMode,
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

  if (!SUPPORTED_KV_CACHE_MODES.includes(manifest.runtime?.kvCacheMode)) {
    errors.push("runtime.kvCacheMode must be stateful-preferred, slot-ring, or contiguous-sliding");
  }

  validateRuntimeGraphSchema(manifest.runtime?.graphSchema, errors);
}

function validateRuntimeGraphSchema(graphSchema, errors) {
  if (!isRecord(graphSchema)) {
    errors.push("runtime.graphSchema must describe Core ML prefill/decode IO");
    return;
  }

  for (const [field, expected] of Object.entries({
    layerCount: EXPECTED_GRAPH_SCHEMA.layerCount,
    kvHeads: EXPECTED_GRAPH_SCHEMA.kvHeads,
    headDimension: EXPECTED_GRAPH_SCHEMA.headDimension
  })) {
    if (graphSchema[field] !== expected) {
      errors.push(`runtime.graphSchema.${field} must be ${expected}`);
    }
  }
  if (!SUPPORTED_GRAPH_INTERFACES.includes(graphSchema.interface)) {
    errors.push("runtime.graphSchema.interface must be logits-layered-kv, stateful-kv, or stateful-step-kv");
  }

  validateNamedSchema("runtime.graphSchema.prefill", graphSchema.prefill, EXPECTED_GRAPH_SCHEMA.prefill, errors);
  validateNamedSchema("runtime.graphSchema.decode", graphSchema.decode, expectedDecodeSchema(graphSchema.interface), errors);
}

function expectedDecodeSchema(graphInterface) {
  if (graphInterface === "stateful-step-kv") {
    return {
      ...EXPECTED_GRAPH_SCHEMA.decode,
      tokenID: "input_ids",
      positionID: "position_ids"
    };
  }
  return EXPECTED_GRAPH_SCHEMA.decode;
}

function validateNamedSchema(path, schema, expectedSchema, errors) {
  if (!isRecord(schema)) {
    errors.push(`${path} must be an object`);
    return;
  }

  for (const [field, expected] of Object.entries(expectedSchema)) {
    if (schema[field] !== expected) {
      errors.push(`${path}.${field} must be ${expected}`);
    }
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

  for (const field of ["prefillSHA256", "decodeSHA256", "tokenizerSHA256"]) {
    if (field in (manifest.asset ?? {}) && !isSHA256(manifest.asset[field])) {
      errors.push(`asset.${field} must be a 64-character hex digest`);
    }
  }

  validateStatefulSharedArtifacts(manifest, errors);
  validateAssetVariants(manifest, errors);
}

function validateStatefulSharedArtifacts(manifest, errors) {
  if (!STATEFUL_GRAPH_INTERFACES.includes(manifest.runtime?.graphSchema?.interface)) {
    return;
  }

  if (manifest.asset?.prefillPath !== manifest.asset?.decodePath) {
    errors.push("stateful Core ML graphs must use the same artifact path for prefill and decode");
  }

  const variants = manifest.asset?.variants;
  if (!isRecord(variants)) {
    return;
  }

  for (const [contextVariant, variant] of Object.entries(variants)) {
    if (variant.prefillPath !== variant.decodePath) {
      errors.push(`asset.variants.${contextVariant} must use the same artifact path for prefill and decode for stateful Core ML graphs`);
    }
  }
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

    if (typeof (variant.tokenizerPath ?? manifest.asset?.tokenizerPath) !== "string") {
      errors.push(`asset.variants.${contextVariant}.tokenizerPath must be present`);
    }

    for (const field of ["prefillSHA256", "decodeSHA256", "tokenizerSHA256"]) {
      const value = variant[field] ?? manifest.asset?.[field];
      if (!isSHA256(value)) {
        errors.push(`asset.variants.${contextVariant}.${field} must be a 64-character hex digest`);
      }
    }
  }
}

function validateQuantization(manifest, errors) {
  const quantization = manifest.quantization ?? {};
  if (quantization.strategy !== "mixed-precision-fidelity-first") {
    errors.push("quantization.strategy must be mixed-precision-fidelity-first");
  }

  if (!["fp16", "int8"].includes(quantization.kvCache)) {
    errors.push("quantization.kvCache must be fp16 or int8");
  }

  if (quantization.structuralReduction !== false) {
    errors.push("structuralReduction must be false");
  }

  if (!isRecord(quantization.weights)) {
    errors.push("quantization.weights must describe per-component precision");
    return;
  }

  for (const component of QUANTIZED_WEIGHT_COMPONENTS) {
    if (typeof quantization.weights[component] !== "string") {
      errors.push(`quantization.weights.${component} must be present`);
    } else if (!SUPPORTED_PRECISIONS.includes(quantization.weights[component])) {
      errors.push(`quantization.weights.${component} must be fp16, int8, or int4`);
    }
  }

  validateLayerOverrides(quantization, manifest.architecture?.layers, errors);
}

function validateLayerOverrides(quantization, layerCount, errors) {
  if (quantization.layerOverrides === undefined) {
    return;
  }
  if (!isRecord(quantization.layerOverrides)) {
    errors.push("quantization.layerOverrides must be an object");
    return;
  }

  const resolvedLayerCount = Number.isInteger(layerCount) ? layerCount : EXPECTED_ARCHITECTURE.layers;
  for (const [component, overrides] of Object.entries(quantization.layerOverrides)) {
    if (!TRANSFORMER_WEIGHT_COMPONENTS.includes(component)) {
      errors.push(`quantization.layerOverrides.${component} is not supported`);
      continue;
    }
    if (!isRecord(overrides)) {
      errors.push(`quantization.layerOverrides.${component} must be an object`);
      continue;
    }

    for (const [rawLayer, precision] of Object.entries(overrides)) {
      const layer = Number(rawLayer);
      if (!Number.isInteger(layer)) {
        errors.push(`quantization.layerOverrides.${component} layer must be an integer`);
        continue;
      }
      if (layer < 0 || layer >= resolvedLayerCount) {
        errors.push(`quantization.layerOverrides.${component}.${layer} is outside layer count`);
      }
      if (!SUPPORTED_PRECISIONS.includes(precision)) {
        errors.push(`quantization.layerOverrides.${component}.${layer} must be fp16, int8, or int4`);
      }
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

function isSHA256(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}
