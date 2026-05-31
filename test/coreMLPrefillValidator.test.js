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
const validatorScript = path.join(repoRoot, "tools", "validation", "validate-coreml-prefill.py");

test("Core ML prefill validator exposes dtype and top-k gate options", async () => {
  const { stdout } = await execFileAsync(python, [validatorScript, "--help"], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.match(stdout, /--torch-dtype/);
  assert.match(stdout, /--minimum-top-k-agreement/);
  assert.match(stdout, /--require-top1-match/);
  assert.match(stdout, /--maximum-mean-absolute-error/);
});

test("Core ML prefill validator gate reports failed top-k alignment", async () => {
  const { stdout } = await execFileAsync(python, ["-c", `
import importlib.util
from pathlib import Path
from types import SimpleNamespace

script = Path("tools/validation/validate-coreml-prefill.py").resolve()
spec = importlib.util.spec_from_file_location("validate_coreml_prefill", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

args = SimpleNamespace(
    minimum_top_k_agreement=10,
    require_top1_match=True,
    maximum_mean_absolute_error=0.1,
)
report = {
    "topKAgreement": 9,
    "top1Matches": True,
    "meanAbsoluteError": 0.05,
}
gate = module.evaluate_gate(report, args)
assert not gate["ok"]
assert "top-k agreement 9 is below 10" in gate["failures"]

report["topKAgreement"] = 10
gate = module.evaluate_gate(report, args)
assert gate["ok"]
assert gate["failures"] == []
print("ok")
`], {
    cwd: repoRoot,
    maxBuffer: 1024 * 1024
  });

  assert.equal(stdout.trim(), "ok");
});
