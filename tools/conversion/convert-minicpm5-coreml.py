#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "hf" / "MiniCPM5-1B"
DEFAULT_PROMPT = "Apple Watch local inference test."


class MiniCPMPrefillWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        ids = input_ids.to(torch.long)
        positions = position_ids.to(torch.long)
        output = self.model(
            input_ids=ids,
            attention_mask=causal_mask,
            position_ids=positions,
            use_cache=False,
        )
        return output.logits[:, -1, :]


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "conversion-report.json"
    report: dict[str, Any] = {
        "modelId": args.model_id,
        "contextTokens": args.context_tokens,
        "computePrecision": args.compute_precision,
        "quantize": args.quantize,
        "prompt": args.prompt,
        "stages": [],
    }

    try:
        snapshot_path = run_stage(report, "download_snapshot", lambda: download_snapshot(args))
        tokenizer = run_stage(report, "load_tokenizer", lambda: load_tokenizer(snapshot_path))
        model = run_stage(report, "load_model", lambda: load_model(snapshot_path))
        example_inputs = run_stage(
            report,
            "build_example_inputs",
            lambda: build_example_inputs(tokenizer, args.context_tokens, args.prompt),
        )
        traced = run_stage(report, "trace_prefill", lambda: trace_prefill(model, example_inputs))
        mlpackage_path = run_stage(
            report,
            "convert_prefill_coreml",
            lambda: convert_prefill(traced, example_inputs, output_dir, args),
        )

        if args.quantize:
            mlpackage_path = run_stage(
                report,
                "quantize_coreml_weights",
                lambda: quantize_coreml_package(mlpackage_path, output_dir),
            )

        report["status"] = "succeeded"
        report["mlpackagePath"] = str(mlpackage_path.relative_to(ROOT))
        report["mlpackageBytes"] = directory_size(mlpackage_path)
        print(json.dumps(report, indent=2))
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(report, indent=2))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real MiniCPM5 prefill to Core ML.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="artifacts/coreml/real-minicpm5-prefill-16")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--quantize", action="store_true")
    return parser.parse_args()


def run_stage(report: dict[str, Any], name: str, action):
    started = time.time()
    stage: dict[str, Any] = {"name": name, "status": "running"}
    report["stages"].append(stage)
    try:
        result = action()
        stage["status"] = "succeeded"
        stage["elapsedSeconds"] = round(time.time() - started, 3)
        return result
    except Exception as error:
        stage["status"] = "failed"
        stage["elapsedSeconds"] = round(time.time() - started, 3)
        stage["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
        raise


def download_snapshot(args: argparse.Namespace) -> Path:
    path = snapshot_download(
        repo_id=args.model_id,
        local_dir=args.cache_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
    )
    return Path(path)


def load_tokenizer(snapshot_path: Path):
    return AutoTokenizer.from_pretrained(snapshot_path)


def load_model(snapshot_path: Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def build_example_inputs(
    tokenizer,
    context_tokens: int,
    prompt: str = DEFAULT_PROMPT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(torch.int32)
    token_mask = torch.ones_like(input_ids, dtype=torch.int32)
    if input_ids.shape[1] > context_tokens:
        input_ids = input_ids[:, -context_tokens:]
        token_mask = token_mask[:, -context_tokens:]
    if input_ids.shape[1] < context_tokens:
        padding_tokens = context_tokens - input_ids.shape[1]
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id or 0
        padding = torch.full(
            (1, padding_tokens),
            int(pad_token_id),
            dtype=torch.int32,
        )
        mask_padding = torch.zeros((1, padding_tokens), dtype=torch.int32)
        input_ids = torch.cat([padding, input_ids], dim=1)
        token_mask = torch.cat([mask_padding, token_mask], dim=1)

    position_ids = (token_mask.cumsum(dim=-1) - 1).clamp(min=0).to(torch.int32)
    causal_mask = build_4d_causal_mask(token_mask)
    return input_ids, position_ids, causal_mask


def build_4d_causal_mask(token_mask: torch.Tensor) -> torch.Tensor:
    context_tokens = token_mask.shape[-1]
    key_is_real = token_mask[:, None, None, :].to(torch.bool)
    causal = torch.tril(torch.ones((context_tokens, context_tokens), dtype=torch.bool))
    allowed = causal[None, None, :, :] & key_is_real
    blocked = torch.full((1, 1, context_tokens, context_tokens), torch.finfo(torch.float16).min)
    return torch.where(allowed, torch.zeros_like(blocked), blocked).to(torch.float16)


def trace_prefill(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.jit.ScriptModule:
    wrapper = MiniCPMPrefillWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        _ = wrapper(*example_inputs)
        return torch.jit.trace(wrapper, example_inputs, strict=False)


def convert_prefill(
    traced: torch.jit.ScriptModule,
    example_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    package_path = output_dir / f"prefill-{args.context_tokens}.mlpackage"
    if package_path.exists():
        shutil.rmtree(package_path)

    precision = ct.precision.FLOAT16 if args.compute_precision == "float16" else ct.precision.FLOAT32
    input_ids, position_ids, causal_mask = example_inputs
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.watchOS10,
        compute_precision=precision,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=tuple(input_ids.shape),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="position_ids",
                shape=tuple(position_ids.shape),
                dtype=np.int32,
            ),
            ct.TensorType(
                name="causal_mask",
                shape=tuple(causal_mask.shape),
                dtype=np.float16,
            ),
        ],
        outputs=[ct.TensorType(name="logits")],
    )
    mlmodel.save(package_path)
    return package_path


def quantize_coreml_package(mlpackage_path: Path, output_dir: Path) -> Path:
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

    quantized_path = output_dir / (mlpackage_path.stem + "-int8.mlpackage")
    if quantized_path.exists():
        shutil.rmtree(quantized_path)

    model = ct.models.MLModel(str(mlpackage_path))
    config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric"))
    quantized = linear_quantize_weights(model, config=config)
    quantized.save(str(quantized_path))
    return quantized_path


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
