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

    tokenizer = conversion.load_tokenizer(snapshot_path)
    model = conversion.load_model(snapshot_path)
    input_ids, token_mask, position_ids, causal_mask = conversion.build_prefill_tensors(
        tokenizer,
        args.context_tokens,
        args.prompt,
    )
    example_inputs = conversion.build_decode_inputs(model, input_ids, token_mask, position_ids, causal_mask)

    torch_outputs, torch_seconds = run_torch(model, conversion, example_inputs)
    coreml_outputs, coreml_seconds = run_coreml(Path(args.mlpackage).resolve(), example_inputs, model.config.num_hidden_layers)
    report = build_report(args, torch_outputs, coreml_outputs, torch_seconds, coreml_seconds, model.config.num_hidden_layers)
    print(json.dumps(report, indent=2))

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Core ML decode logits/KV against PyTorch MiniCPM.")
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


def run_torch(model: torch.nn.Module, conversion, example_inputs) -> tuple[tuple[np.ndarray, ...], float]:
    wrapper = conversion.MiniCPMDecodeKVWrapper(model, model.config.num_hidden_layers)
    wrapper.eval()
    started = time.perf_counter()
    with torch.no_grad():
        outputs = wrapper(*example_inputs)
    arrays = tuple(output.detach().cpu().float().numpy() for output in outputs)
    return arrays, time.perf_counter() - started


def run_coreml(
    mlpackage_path: Path,
    example_inputs,
    num_hidden_layers: int,
) -> tuple[dict[str, np.ndarray], float]:
    token_id, position_id, causal_mask, *past_key_values = example_inputs
    model = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    inputs = {
        "token_id": token_id.cpu().numpy().astype(np.int32),
        "position_id": position_id.cpu().numpy().astype(np.int32),
        "causal_mask": causal_mask.cpu().numpy().astype(np.float16),
    }
    for index in range(num_hidden_layers):
        inputs[f"past_key_{index}"] = past_key_values[index * 2].cpu().numpy().astype(np.float16)
        inputs[f"past_value_{index}"] = past_key_values[index * 2 + 1].cpu().numpy().astype(np.float16)

    started = time.perf_counter()
    prediction = model.predict(inputs)
    arrays = {name: value.astype(np.float32) for name, value in prediction.items()}
    return arrays, time.perf_counter() - started


def build_report(
    args: argparse.Namespace,
    torch_outputs: tuple[np.ndarray, ...],
    coreml_outputs: dict[str, np.ndarray],
    torch_seconds: float,
    coreml_seconds: float,
    num_hidden_layers: int,
) -> dict[str, Any]:
    torch_logits = torch_outputs[0]
    coreml_logits = coreml_outputs["logits"]
    diff = np.abs(torch_logits - coreml_logits)
    torch_top = top_indices(torch_logits, args.top_k)
    coreml_top = top_indices(coreml_logits, args.top_k)
    kv_max_error = 0.0
    kv_mean_errors = []
    for index in range(num_hidden_layers):
        torch_key = torch_outputs[index * 2 + 1]
        torch_value = torch_outputs[index * 2 + 2]
        coreml_key = coreml_outputs[f"new_key_{index}"]
        coreml_value = coreml_outputs[f"new_value_{index}"]
        key_diff = np.abs(torch_key - coreml_key)
        value_diff = np.abs(torch_value - coreml_value)
        kv_max_error = max(kv_max_error, float(key_diff.max()), float(value_diff.max()))
        kv_mean_errors.extend([float(key_diff.mean()), float(value_diff.mean())])

    return {
        "mlpackage": args.mlpackage,
        "contextTokens": args.context_tokens,
        "prompt": args.prompt,
        "torchSeconds": round(torch_seconds, 6),
        "coremlCPUSeconds": round(coreml_seconds, 6),
        "logitsMaxAbsoluteError": float(diff.max()),
        "logitsMeanAbsoluteError": float(diff.mean()),
        "kvMaxAbsoluteError": kv_max_error,
        "kvMeanAbsoluteError": float(np.mean(kv_mean_errors)),
        "topK": args.top_k,
        "torchTopK": torch_top,
        "coremlTopK": coreml_top,
        "topKAgreement": len(set(torch_top).intersection(coreml_top)),
        "top1Matches": bool(torch_top[0] == coreml_top[0]),
    }


def top_indices(logits: np.ndarray, top_k: int) -> list[int]:
    row = logits.reshape(-1)
    indices = np.argpartition(row, -top_k)[-top_k:]
    return [int(index) for index in indices[np.argsort(row[indices])[::-1]]]


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
