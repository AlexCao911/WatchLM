#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CONVERSION_SCRIPT = ROOT / "tools" / "conversion" / "convert-minicpm5-coreml.py"


def main() -> None:
    args = parse_args()
    conversion = load_conversion_module()
    snapshot_path = Path(args.cache_dir).resolve()
    report_path = Path(args.report).resolve() if args.report else None

    tokenizer = conversion.load_tokenizer(snapshot_path)
    model = conversion.load_model(snapshot_path, args.torch_dtype)
    input_ids, _, position_ids, causal_mask = conversion.build_prefill_tensors(
        tokenizer,
        args.context_tokens,
        args.prompt,
    )
    example_inputs = (input_ids, position_ids, causal_mask)

    torch_logits, torch_seconds = run_torch(model, conversion, example_inputs, args.graph)
    coreml_logits, coreml_seconds = run_coreml(
        Path(args.mlpackage).resolve(),
        example_inputs,
        compute_units(args.compute_units),
    )

    report = build_report(args, torch_logits, coreml_logits, torch_seconds, coreml_seconds)
    report["gate"] = evaluate_gate(report, args)
    print(json.dumps(report, indent=2))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not report["gate"]["ok"]:
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Core ML prefill logits against PyTorch causal LM teacher.")
    parser.add_argument("--mlpackage", required=True)
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts" / "hf" / "MiniCPM5-1B"))
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--graph", choices=["prefill", "prefill-kv"], default="prefill")
    parser.add_argument("--compute-units", choices=["all", "cpu"], default="cpu")
    parser.add_argument("--torch-dtype", choices=["float16", "float32", "bfloat16", "auto"], default="float16")
    parser.add_argument("--prompt", default="Apple Watch local inference test.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--minimum-top-k-agreement", type=int, default=0)
    parser.add_argument("--require-top1-match", action="store_true")
    parser.add_argument("--maximum-mean-absolute-error", type=float)
    parser.add_argument("--report")
    return parser.parse_args()


def load_conversion_module():
    spec = importlib.util.spec_from_file_location("watchlm_minicpm_conversion", CONVERSION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load conversion script: {CONVERSION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_torch(model: torch.nn.Module, conversion, example_inputs, graph: str) -> tuple[np.ndarray, float]:
    if graph == "prefill-kv":
        wrapper = conversion.MiniCPMPrefillKVWrapper(model)
    else:
        wrapper = conversion.MiniCPMPrefillWrapper(model)
    wrapper.eval()
    started = time.perf_counter()
    with torch.no_grad():
        output = wrapper(*example_inputs)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.detach().cpu().float().numpy()
    return logits, time.perf_counter() - started


def compute_units(value: str):
    if value == "all":
        return ct.ComputeUnit.ALL
    return ct.ComputeUnit.CPU_ONLY


def run_coreml(mlpackage_path: Path, example_inputs, compute_unit) -> tuple[np.ndarray, float]:
    input_ids, position_ids, causal_mask = example_inputs
    mlmodel = ct.models.MLModel(str(mlpackage_path), compute_units=compute_unit)
    inputs = {
        "input_ids": input_ids.cpu().numpy().astype(np.int32),
        "position_ids": position_ids.cpu().numpy().astype(np.int32),
        "causal_mask": causal_mask.cpu().numpy().astype(np.float16),
    }
    started = time.perf_counter()
    prediction = mlmodel.predict(inputs)
    logits = prediction["logits"].astype(np.float32)
    return logits, time.perf_counter() - started


def build_report(
    args: argparse.Namespace,
    torch_logits: np.ndarray,
    coreml_logits: np.ndarray,
    torch_seconds: float,
    coreml_seconds: float,
) -> dict[str, Any]:
    diff = np.abs(torch_logits - coreml_logits)
    top_k = args.top_k
    torch_top = top_indices(torch_logits, top_k)
    coreml_top = top_indices(coreml_logits, top_k)
    shared_top = sorted(set(torch_top).intersection(coreml_top))

    return {
        "mlpackage": args.mlpackage,
        "graph": args.graph,
        "computeUnits": args.compute_units,
        "contextTokens": args.context_tokens,
        "prompt": args.prompt,
        "torchSeconds": round(torch_seconds, 6),
        "coremlCPUSeconds": round(coreml_seconds, 6),
        "maxAbsoluteError": float(diff.max()),
        "meanAbsoluteError": float(diff.mean()),
        "topK": top_k,
        "torchTopK": torch_top,
        "coremlTopK": coreml_top,
        "topKAgreement": len(shared_top),
        "top1Matches": bool(torch_top[0] == coreml_top[0]),
    }


def evaluate_gate(report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    minimum_top_k_agreement = int(args.minimum_top_k_agreement)
    if minimum_top_k_agreement > 0 and report["topKAgreement"] < minimum_top_k_agreement:
        failures.append(f"top-k agreement {report['topKAgreement']} is below {minimum_top_k_agreement}")

    if args.require_top1_match and not report["top1Matches"]:
        failures.append("top-1 token does not match")

    maximum_mean_absolute_error = args.maximum_mean_absolute_error
    if (
        maximum_mean_absolute_error is not None
        and report["meanAbsoluteError"] > maximum_mean_absolute_error
    ):
        failures.append(
            f"mean absolute error {report['meanAbsoluteError']} exceeds {maximum_mean_absolute_error}"
        )

    return {
        "ok": not failures,
        "minimumTopKAgreement": minimum_top_k_agreement,
        "requireTop1Match": bool(args.require_top1_match),
        "maximumMeanAbsoluteError": maximum_mean_absolute_error,
        "failures": failures,
    }


def top_indices(logits: np.ndarray, top_k: int) -> list[int]:
    row = logits.reshape(-1)
    indices = np.argpartition(row, -top_k)[-top_k:]
    return [int(index) for index in indices[np.argsort(row[indices])[::-1]]]


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
