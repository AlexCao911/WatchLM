import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  EXPECTED_ARCHITECTURE,
  EXPECTED_GRAPH_SCHEMA,
  EXPECTED_MODEL_ID,
  EXPECTED_RUNTIME,
  SUPPORTED_CONTEXT_VARIANTS,
  SUPPORTED_KV_CACHE_MODES,
  assertValidModelManifest,
  selectContextVariant,
  selectModelArtifact,
  summarizeModelManifest,
  validateModelManifest
} from "../tools/validation/modelManifest.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(
  __dirname,
  "..",
  "tools",
  "validation",
  "fixtures",
  "sample-model-manifest.json"
);
const validManifest = JSON.parse(await readFile(fixturePath, "utf8"));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

test("valid MiniCPM5 Core ML manifest passes validation", () => {
  const result = validateModelManifest(validManifest);

  assert.equal(result.ok, true);
  assert.deepEqual(result.errors, []);
  assert.doesNotThrow(() => assertValidModelManifest(validManifest));
});

test("manifest constants encode the fidelity-first MiniCPM5 contract", () => {
  assert.equal(EXPECTED_MODEL_ID, "openbmb/MiniCPM5-1B");
  assert.equal(EXPECTED_RUNTIME, "coreml-mlprogram");
  assert.deepEqual(SUPPORTED_KV_CACHE_MODES, [
    "stateful-preferred",
    "slot-ring",
    "contiguous-sliding"
  ]);
  assert.deepEqual(EXPECTED_GRAPH_SCHEMA, {
    interface: "logits-layered-kv",
    layerCount: 24,
    kvHeads: 2,
    headDimension: 128,
    prefill: {
      inputIDs: "input_ids",
      positionIDs: "position_ids",
      causalMask: "causal_mask",
      logits: "logits",
      keyPrefix: "present_key_",
      valuePrefix: "present_value_"
    },
    decode: {
      tokenID: "token_id",
      positionID: "position_id",
      causalMask: "causal_mask",
      logits: "logits",
      pastKeyPrefix: "past_key_",
      pastValuePrefix: "past_value_",
      newKeyPrefix: "new_key_",
      newValuePrefix: "new_value_"
    }
  });
  assert.deepEqual(SUPPORTED_CONTEXT_VARIANTS, [256, 512, 1024]);
  assert.deepEqual(EXPECTED_ARCHITECTURE, {
    layers: 24,
    hiddenSize: 1536,
    queryHeads: 16,
    kvHeads: 2,
    tokenizerSource: "openbmb/MiniCPM5-1B"
  });
});

test("runtime must be Core ML mlprogram", () => {
  const manifest = clone(validManifest);
  manifest.runtime.type = "llama.cpp";

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /runtime\.type must be coreml-mlprogram/);
});

test("runtime graph schema must expose logits and layered KV IO names", () => {
  const manifest = clone(validManifest);
  manifest.runtime.graphSchema.interface = "next-token";
  manifest.runtime.graphSchema.layerCount = 1;
  manifest.runtime.graphSchema.prefill.logits = "next_token";
  manifest.runtime.graphSchema.decode.pastKeyPrefix = "cache_key_";

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /runtime\.graphSchema\.interface must be logits-layered-kv/);
  assert.match(result.errors.join("\n"), /runtime\.graphSchema\.layerCount must be 24/);
  assert.match(result.errors.join("\n"), /runtime\.graphSchema\.prefill\.logits must be logits/);
  assert.match(result.errors.join("\n"), /runtime\.graphSchema\.decode\.pastKeyPrefix must be past_key_/);
});

test("runtime KV cache mode must stay in the explicit Swift update strategy set", () => {
  const manifest = clone(validManifest);
  manifest.runtime.kvCacheMode = "copy-everything";

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(
    result.errors.join("\n"),
    /runtime\.kvCacheMode must be stateful-preferred, slot-ring, or contiguous-sliding/
  );
});

test("source model must stay MiniCPM5-1B", () => {
  const manifest = clone(validManifest);
  manifest.model.id = "openbmb/OtherModel";

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /model\.id must be openbmb\/MiniCPM5-1B/);
});

test("architecture must preserve MiniCPM5 dimensions and tokenizer", () => {
  const manifest = clone(validManifest);
  manifest.architecture.layers = 23;
  manifest.architecture.hiddenSize = 1024;
  manifest.architecture.queryHeads = 8;
  manifest.architecture.kvHeads = 4;
  manifest.architecture.tokenizer.preserved = false;
  manifest.architecture.tokenizer.vocabularyPreserved = false;

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /architecture\.layers must be 24/);
  assert.match(result.errors.join("\n"), /architecture\.hiddenSize must be 1536/);
  assert.match(result.errors.join("\n"), /architecture\.queryHeads must be 16/);
  assert.match(result.errors.join("\n"), /architecture\.kvHeads must be 2/);
  assert.match(result.errors.join("\n"), /tokenizer must be preserved/);
  assert.match(result.errors.join("\n"), /vocabulary must be preserved/);
});

test("context variants must be supported Apple Watch SE sizes", () => {
  const manifest = clone(validManifest);
  manifest.contextVariants = [256, 768, 1024];

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /unsupported context variant 768/);
});

test("model assets must stay outside the watch app bundle", () => {
  const manifest = clone(validManifest);
  manifest.asset.storage = "app-bundle";

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /asset\.storage must not be app-bundle/);
});

test("model assets must include SE 2 and SE 3 quantized variants", () => {
  const manifest = clone(validManifest);
  delete manifest.asset.variants["256"];

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /asset\.variants\.256 must be present for watch-se-2/);
});

test("quantization policy allows fp16 KV cache for fidelity profiles", () => {
  const manifest = clone(validManifest);
  manifest.quantization.kvCache = "fp16";

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, true);
});

test("quantization policy must be mixed precision with supported KV cache", () => {
  const manifest = clone(validManifest);
  manifest.quantization.strategy = "uniform-int4";
  manifest.quantization.kvCache = "int4";
  manifest.quantization.structuralReduction = true;

  const result = validateModelManifest(manifest);

  assert.equal(result.ok, false);
  assert.match(result.errors.join("\n"), /quantization\.strategy must be mixed-precision-fidelity-first/);
  assert.match(result.errors.join("\n"), /quantization\.kvCache must be fp16 or int8/);
  assert.match(result.errors.join("\n"), /structuralReduction must be false/);
});

test("selectContextVariant clamps to the largest supported variant that fits", () => {
  assert.equal(selectContextVariant(validManifest, "watch-se-3", 1024), 1024);
  assert.equal(selectContextVariant(validManifest, "watch-se-3", 999), 512);
  assert.equal(selectContextVariant(validManifest, "watch-se-3", 600), 512);
  assert.equal(selectContextVariant(validManifest, "watch-se-3", 128), 256);
  assert.equal(selectContextVariant(validManifest, "watch-se-2"), 256);
});

test("selectModelArtifact resolves SE 2 and SE 3 prefill/decode variants", () => {
  assert.deepEqual(selectModelArtifact(validManifest, "watch-se-2"), {
    contextVariant: 256,
    deviceProfile: "watch-se-2",
    prefillPath: "Models/MiniCPM5/prefill-256.mlpackage",
    decodePath: "Models/MiniCPM5/decode-256.mlpackage",
    tokenizerPath: "Models/MiniCPM5/tokenizer.json",
    sha256: "1111111111111111111111111111111111111111111111111111111111111111",
    prefillSHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    decodeSHA256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    tokenizerSHA256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  });
  assert.deepEqual(selectModelArtifact(validManifest, "watch-se-3"), {
    contextVariant: 512,
    deviceProfile: "watch-se-3",
    prefillPath: "Models/MiniCPM5/prefill-512.mlpackage",
    decodePath: "Models/MiniCPM5/decode-512.mlpackage",
    tokenizerPath: "Models/MiniCPM5/tokenizer.json",
    sha256: "2222222222222222222222222222222222222222222222222222222222222222",
    prefillSHA256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    decodeSHA256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
    tokenizerSHA256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
  });
});

test("assertValidModelManifest throws one combined validation error", () => {
  const manifest = clone(validManifest);
  manifest.model.id = "wrong";
  manifest.runtime.type = "wrong";

  assert.throws(
    () => assertValidModelManifest(manifest),
    /Invalid model manifest:\n- model\.id must be openbmb\/MiniCPM5-1B\n- runtime\.type must be coreml-mlprogram/
  );
});

test("summarizeModelManifest exposes audit-friendly details", () => {
  const summary = summarizeModelManifest(validManifest);

  assert.deepEqual(summary, {
    modelId: "openbmb/MiniCPM5-1B",
    runtime: "coreml-mlprogram",
    deviceProfiles: ["watch-se-2", "watch-se-3"],
    contextVariants: [256, 512, 1024],
    assetStorage: "application-support",
    assetVariants: [256, 512],
    kvCacheMode: "stateful-preferred",
    quantizationStrategy: "mixed-precision-fidelity-first",
    kvCachePrecision: "int8"
  });
});
