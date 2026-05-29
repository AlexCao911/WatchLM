import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.join(__dirname, "..");
const python = path.join(repoRoot, ".venv", "bin", "python");
const conversionScript = path.join(repoRoot, "tools", "conversion", "convert-minicpm5-coreml.py");

test("real MiniCPM conversion CLI exposes compression and graph choices", async () => {
  const { stdout } = await execFileAsync(python, [conversionScript, "--help"], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.match(stdout, /--compression/);
  assert.match(stdout, /none/);
  assert.match(stdout, /int8/);
  assert.match(stdout, /int4/);
  assert.match(stdout, /--graph/);
  assert.match(stdout, /prefill/);
  assert.match(stdout, /prefill-kv/);
  assert.match(stdout, /decode/);
});
