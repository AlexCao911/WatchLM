#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    model = conversion.load_model(snapshot_path)
    input_ids, _, position_ids, causal_mask = conversion.build_prefill_tensors(
        tokenizer,
        args.context_tokens,
        args.prompt,
    )
    example_inputs = (input_ids, position_ids, causal_mask)

    torch_logits, torch_seconds = run_torch(model, conversion, example_inputs)
    coreml_logits, coreml_seconds = run_coreml(Path(args.mlpackage).resolve(), example_inputs)

    report = build_report(args, torch_logits, coreml_logits, torch_seconds, coreml_seconds)
    print(json.dumps(report, indent=2))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Core ML prefill logits against PyTorch MiniCPM.")
    parser.add_argument("--mlpackage", required=True)
    parser.add_argument("--cache-dir", default=str(ROOT / "artifacts" / "hf" / "MiniCPM5-1B"))
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="Apple Watch local inference test.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report")
    return parser.parse_args()


def load_conversion_module():
    spec = importlib.util.spec_from_file_location("watchlm_minicpm_conversion", CONVERSION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load conversion script: {CONVERSION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_torch(model: torch.nn.Module, conversion, example_inputs) -> tuple[np.ndarray, float]:
    wrapper = conversion.MiniCPMPrefillWrapper(model)
    wrapper.eval()
    started = time.perf_counter()
    with torch.no_grad():
        logits = wrapper(*example_inputs).detach().cpu().float().numpy()
    return logits, time.perf_counter() - started


def run_coreml(mlpackage_path: Path, example_inputs) -> tuple[np.ndarray, float]:
    input_ids, position_ids, causal_mask = example_inputs
    mlmodel = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)
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


def top_indices(logits: np.ndarray, top_k: int) -> list[int]:
    row = logits.reshape(-1)
    indices = np.argpartition(row, -top_k)[-top_k:]
    return [int(index) for index in indices[np.argsort(row[indices])[::-1]]]


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
