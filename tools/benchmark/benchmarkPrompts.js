import { readFile } from "node:fs/promises";

export const REQUIRED_PROMPT_CATEGORIES = Object.freeze([
  "zh_short_instruction",
  "en_short_instruction",
  "code_small_fix",
  "watch_utility",
  "safety_refusal"
]);

const MAX_SMOKE_PROMPT_TOKENS = 256;
const MIN_NEW_TOKENS = 16;
const MAX_NEW_TOKENS = 96;

export async function loadBenchmarkPrompts(fileUrlOrPath) {
  const raw = await readFile(fileUrlOrPath, "utf8");
  const parsed = JSON.parse(raw);
  const prompts = Array.isArray(parsed) ? parsed : parsed.prompts;

  const result = validateBenchmarkPrompts(prompts);
  if (!result.ok) {
    throw new Error(`Invalid benchmark prompts:\n- ${result.errors.join("\n- ")}`);
  }

  return prompts;
}

export function validateBenchmarkPrompts(prompts) {
  const errors = [];

  if (!Array.isArray(prompts)) {
    return {
      ok: false,
      errors: ["prompts must be an array"]
    };
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

    if (!REQUIRED_PROMPT_CATEGORIES.includes(prompt.category)) {
      errors.push(`prompt[${index}].category is unsupported`);
    } else {
      seenCategories.add(prompt.category);
    }

    if (typeof prompt.language !== "string" || prompt.language.trim() === "") {
      errors.push(`prompt[${index}].language must be a non-empty string`);
    }

    if (typeof prompt.input !== "string" || prompt.input.trim() === "") {
      errors.push(`prompt[${index}].input must be a non-empty string`);
    } else if (estimatePromptTokens(prompt) > MAX_SMOKE_PROMPT_TOKENS) {
      errors.push(`prompt[${index}].input must fit the 256 token smoke baseline`);
    }

    if (
      !Number.isInteger(prompt.maxNewTokens) ||
      prompt.maxNewTokens < MIN_NEW_TOKENS ||
      prompt.maxNewTokens > MAX_NEW_TOKENS
    ) {
      errors.push(`prompt[${index}].maxNewTokens must be between 16 and 96`);
    }

    if (!Array.isArray(prompt.qualityChecks) || prompt.qualityChecks.length === 0) {
      errors.push(`prompt[${index}].qualityChecks must be a non-empty array`);
    }
  });

  for (const category of REQUIRED_PROMPT_CATEGORIES) {
    if (!seenCategories.has(category)) {
      errors.push(`missing required category ${category}`);
    }
  }

  return {
    ok: errors.length === 0,
    errors
  };
}

export function groupPromptsByCategory(prompts) {
  const grouped = new Map(REQUIRED_PROMPT_CATEGORIES.map((category) => [category, []]));

  for (const prompt of prompts) {
    if (grouped.has(prompt.category)) {
      grouped.get(prompt.category).push(prompt);
    }
  }

  return grouped;
}

export function estimatePromptTokens(prompt) {
  const input = typeof prompt === "string" ? prompt : prompt.input ?? "";
  return Math.ceil(Array.from(input).length / 4);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
