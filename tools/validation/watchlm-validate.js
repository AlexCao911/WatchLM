#!/usr/bin/env node
import { readFile } from "node:fs/promises";

import { loadCalibrationPromptSuite } from "../benchmark/calibrationPrompts.js";
import { loadBenchmarkPrompts } from "../benchmark/benchmarkPrompts.js";
import { summarizeBenchmarkReport, validateBenchmarkReport } from "../benchmark/benchmarkReport.js";
import {
  assertValidModelCandidateSuite,
  evaluateModelCandidate,
  recommendModelCandidates,
  summarizeCandidateEvaluation
} from "./modelCandidateSizing.js";
import { assertValidModelManifest, summarizeModelManifest } from "./modelManifest.js";

async function main(argv) {
  const [command, ...args] = argv;

  switch (command) {
    case "manifest":
      await validateManifestCommand(args);
      break;
    case "prompts":
      await validatePromptsCommand(args);
      break;
    case "calibration-prompts":
      await validateCalibrationPromptsCommand(args);
      break;
    case "report":
      await validateReportCommand(args);
      break;
    case "candidates":
      await validateCandidatesCommand(args);
      break;
    case "all":
      await validateAllCommand(args);
      break;
    default:
      throw new Error(usage());
  }
}

async function validateManifestCommand(args) {
  const [manifestPath] = args;
  if (!manifestPath) {
    throw new Error("manifest command requires a path");
  }

  const manifest = await readJson(manifestPath);
  assertValidModelManifest(manifest);
  const summary = summarizeModelManifest(manifest);
  console.log(
    `manifest ok: ${summary.modelId} ${summary.runtime} contexts=${summary.contextVariants.join(",")}`
  );
}

async function validatePromptsCommand(args) {
  const [promptsPath] = args;
  if (!promptsPath) {
    throw new Error("prompts command requires a path");
  }

  const prompts = await loadBenchmarkPrompts(promptsPath);
  console.log(`prompts ok: ${prompts.length} prompts`);
}

async function validateCalibrationPromptsCommand(args) {
  const [promptsPath] = args;
  if (!promptsPath) {
    throw new Error("calibration-prompts command requires a path");
  }

  const suite = await loadCalibrationPromptSuite(promptsPath);
  console.log(
    `calibration prompts ok: ${suite.prompts.length} prompts, prefixes=${suite.prefixTokenCounts.join(",")}`
  );
}

async function validateReportCommand(args) {
  const [reportPath] = args;
  if (!reportPath) {
    throw new Error("report command requires a path");
  }

  const reports = await readReports(reportPath);
  for (const report of reports) {
    const result = validateBenchmarkReport(report);
    if (!result.ok) {
      throw new Error(`Invalid benchmark report:\n- ${result.errors.join("\n- ")}`);
    }
  }

  const summaries = reports.map(summarizeBenchmarkReport);
  const passing = summaries.filter((summary) => summary.gatesPass).length;
  console.log(`report ok: ${reports.length} reports, ${passing} passing gates`);
}

async function validateCandidatesCommand(args) {
  const [candidatesPath] = args;
  if (!candidatesPath) {
    throw new Error("candidates command requires a path");
  }

  const suite = await readJson(candidatesPath);
  assertValidModelCandidateSuite(suite);
  const summaries = suite.candidates
    .map((candidate) => summarizeCandidateEvaluation(evaluateModelCandidate(candidate, "watch-se-2")));
  const passing = summaries.filter((summary) => summary.gatePass).length;
  const recommendations = recommendModelCandidates(suite, "watch-se-2");
  const next = recommendations.find((summary) => summary.recommendation === "convert-next");

  console.log(`candidates ok: ${summaries.length} candidates, ${passing} passing SE2 gate`);
  if (next) {
    console.log(`recommended next: ${next.id} (${next.sourceModelId})`);
  }
  for (const summary of summaries) {
    const status = summary.gatePass ? "pass" : "fail";
    console.log(
      `${summary.id}: ${status} artifact=${summary.artifactMB}MB peakRSS=${summary.peakRSSMB}MB role=${summary.role}`
    );
  }
}

async function validateAllCommand(args) {
  const manifestPath = flagValue(args, "--manifest");
  const promptsPath = flagValue(args, "--prompts");
  const reportPath = flagValue(args, "--report");

  if (!manifestPath || !promptsPath || !reportPath) {
    throw new Error("all command requires --manifest, --prompts, and --report");
  }

  await validateManifestCommand([manifestPath]);
  await validatePromptsCommand([promptsPath]);
  await validateReportCommand([reportPath]);
}

async function readJson(filePath) {
  return JSON.parse(await readFile(filePath, "utf8"));
}

async function readReports(filePath) {
  const parsed = await readJson(filePath);
  return Array.isArray(parsed) ? parsed : parsed.reports;
}

function flagValue(args, flag) {
  const index = args.indexOf(flag);
  return index === -1 ? undefined : args[index + 1];
}

function usage() {
  return [
    "Usage:",
    "  watchlm-validate manifest <path>",
    "  watchlm-validate prompts <path>",
    "  watchlm-validate calibration-prompts <path>",
    "  watchlm-validate report <path>",
    "  watchlm-validate candidates <path>",
    "  watchlm-validate all --manifest <path> --prompts <path> --report <path>"
  ].join("\n");
}

main(process.argv.slice(2)).catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
