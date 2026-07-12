import { readFile } from "node:fs/promises";

export const REQUIRED_CALIBRATION_CATEGORIES = Object.freeze([
  "zh_short_instruction",
  "en_short_instruction",
  "watch_utility",
  "code_small_fix",
  "stop_sequence",
  "safety_refusal"
]);

const EXPECTED_SCHEMA_VERSION = 1;
const EXPECTED_MODEL_ID = "openbmb/MiniCPM5-1B";
const EXPECTED_CONTEXT_TOKENS = 256;
const EXPECTED_PROMPT_FORMAT = "minicpm5-chat-template-no-think";
const MIN_NEW_TOKENS = 1;
const MAX_NEW_TOKENS = 96;
const NO_THINK_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

export async function loadCalibrationPromptSuite(fileUrlOrPath) {
  const raw = await readFile(fileUrlOrPath, "utf8");
  const suite = JSON.parse(raw);
  const result = validateCalibrationPromptSuite(suite);
  if (!result.ok) {
    throw new Error(`Invalid calibration prompts:\n- ${result.errors.join("\n- ")}`);
  }
  return suite;
}

export function validateCalibrationPromptSuite(suite) {
  const errors = [];
  if (!isRecord(suite)) {
    return {
      ok: false,
      errors: ["suite must be an object"]
    };
  }

  if (suite.schemaVersion !== EXPECTED_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${EXPECTED_SCHEMA_VERSION}`);
  }

  if (suite.modelId !== EXPECTED_MODEL_ID) {
    errors.push(`modelId must be ${EXPECTED_MODEL_ID}`);
  }

  if (suite.tokenizerSource !== EXPECTED_MODEL_ID) {
    errors.push(`tokenizerSource must be ${EXPECTED_MODEL_ID}`);
  }

  if (suite.contextTokens !== EXPECTED_CONTEXT_TOKENS) {
    errors.push(`contextTokens must be ${EXPECTED_CONTEXT_TOKENS}`);
  }

  if (suite.promptFormat !== EXPECTED_PROMPT_FORMAT) {
    errors.push(`promptFormat must be ${EXPECTED_PROMPT_FORMAT}`);
  }

  validatePrefixTokenCounts(suite, errors);
  validatePrompts(suite.prompts, errors);

  return {
    ok: errors.length === 0,
    errors
  };
}

export function toBenchmarkPrompts(suite, options = {}) {
  const maxNewTokens = options.maxNewTokens;
  return suite.prompts.map((prompt) => ({
    id: prompt.id,
    category: prompt.category,
    language: prompt.language,
    input: prompt.renderedPrompt,
    maxNewTokens: maxNewTokens ?? prompt.maxNewTokens,
    qualityChecks: prompt.tags
  }));
}

function validatePrefixTokenCounts(suite, errors) {
  if (!Array.isArray(suite.prefixTokenCounts) || suite.prefixTokenCounts.length === 0) {
    errors.push("prefixTokenCounts must be a non-empty array");
    return;
  }

  if (!suite.prefixTokenCounts.every((value) => Number.isInteger(value) && value > 0)) {
    errors.push("prefixTokenCounts must contain positive integers");
  }

  if (
    suite.prefixTokenCounts
      .slice(1)
      .some((value, index) => suite.prefixTokenCounts[index] >= value)
  ) {
    errors.push("prefixTokenCounts must be strictly increasing");
  }

  if (suite.prefixTokenCounts.some((value) => value > suite.contextTokens)) {
    errors.push("prefixTokenCounts must be <= contextTokens");
  }
}

function validatePrompts(prompts, errors) {
  if (!Array.isArray(prompts) || prompts.length === 0) {
    errors.push("prompts must be a non-empty array");
    return;
  }

  const seenIds = new Set();
  const seenCategories = new Set();
  prompts.forEach((prompt, index) => {
    if (!isRecord(prompt)) {
      errors.push(`prompt[${index}] must be an object`);
      return;
    }

    if (typeof prompt.id !== "string" || prompt.id.trim() === "") {
      errors.push(`prompt[${index}].id must be a non-empty string`);
    } else if (seenIds.has(prompt.id)) {
      errors.push(`prompt[${index}].id must be unique`);
    } else {
      seenIds.add(prompt.id);
    }

    if (!REQUIRED_CALIBRATION_CATEGORIES.includes(prompt.category)) {
      errors.push(`prompt[${index}].category is unsupported`);
    } else {
      seenCategories.add(prompt.category);
    }

    if (typeof prompt.language !== "string" || prompt.language.trim() === "") {
      errors.push(`prompt[${index}].language must be a non-empty string`);
    }

    if (!Array.isArray(prompt.messages) || prompt.messages.length === 0) {
      errors.push(`prompt[${index}].messages must be a non-empty array`);
    } else if (prompt.messages.some((message) => !isRecord(message) || typeof message.content !== "string" || message.content.trim() === "")) {
      errors.push(`prompt[${index}].messages must not contain empty content`);
    }

    if (!usesMiniCPMNoThinkTemplate(prompt.renderedPrompt)) {
      errors.push(`prompt[${index}].renderedPrompt must use the MiniCPM no-think assistant prefix`);
    }

    if (
      !Number.isInteger(prompt.maxNewTokens) ||
      prompt.maxNewTokens < MIN_NEW_TOKENS ||
      prompt.maxNewTokens > MAX_NEW_TOKENS
    ) {
      errors.push(`prompt[${index}].maxNewTokens must be between ${MIN_NEW_TOKENS} and ${MAX_NEW_TOKENS}`);
    }

    if (!Array.isArray(prompt.tags) || prompt.tags.length === 0 || prompt.tags.some((tag) => typeof tag !== "string" || tag.trim() === "")) {
      errors.push(`prompt[${index}].tags must be a non-empty array`);
    }
  });

  for (const category of REQUIRED_CALIBRATION_CATEGORIES) {
    if (!seenCategories.has(category)) {
      errors.push(`missing required category ${category}`);
    }
  }
}

function usesMiniCPMNoThinkTemplate(renderedPrompt) {
  return (
    typeof renderedPrompt === "string" &&
    renderedPrompt.startsWith("<s><|im_start|>system\n") &&
    renderedPrompt.includes(NO_THINK_ASSISTANT_PREFIX)
  );
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
