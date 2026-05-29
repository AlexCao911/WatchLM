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
from transformers.cache_utils import DynamicCache
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


class MiniCPMPrefillKVWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ids = input_ids.to(torch.long)
        positions = position_ids.to(torch.long)
        output = self.model(
            input_ids=ids,
            attention_mask=causal_mask,
            position_ids=positions,
            use_cache=True,
        )
        values: list[torch.Tensor] = [output.logits[:, -1, :]]
        for layer in output.past_key_values.layers:
            values.append(layer.keys)
            values.append(layer.values)
        return tuple(values)


class MiniCPMDecodeKVWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, num_hidden_layers: int) -> None:
        super().__init__()
        self.model = model
        self.num_hidden_layers = num_hidden_layers

    def forward(
        self,
        token_id: torch.Tensor,
        position_id: torch.Tensor,
        causal_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pairs = []
        for index in range(self.num_hidden_layers):
            pairs.append((past_key_values[index * 2], past_key_values[index * 2 + 1]))
        cache = DynamicCache(pairs, config=self.model.config)
        output = self.model(
            input_ids=token_id.to(torch.long),
            attention_mask=causal_mask,
            position_ids=position_id.to(torch.long),
            past_key_values=cache,
            use_cache=True,
        )

        values: list[torch.Tensor] = [output.logits[:, -1, :]]
        for layer in output.past_key_values.layers:
            values.append(layer.keys[:, :, -1:, :])
            values.append(layer.values[:, :, -1:, :])
        return tuple(values)


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "conversion-report.json"
    report: dict[str, Any] = {
        "modelId": args.model_id,
        "contextTokens": args.context_tokens,
        "computePrecision": args.compute_precision,
        "graph": args.graph,
        "compression": args.compression,
        "legacyQuantizeFlag": args.quantize,
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
            lambda: build_example_inputs(model, tokenizer, args),
        )
        traced = run_stage(report, f"trace_{args.graph}", lambda: trace_graph(model, example_inputs, args))
        mlpackage_path = run_stage(
            report,
            f"convert_{args.graph}_coreml",
            lambda: convert_graph(traced, example_inputs, output_dir, model.config.num_hidden_layers, args),
        )

        if args.compression != "none":
            mlpackage_path = run_stage(
                report,
                f"compress_coreml_weights_{args.compression}",
                lambda: compress_coreml_package(mlpackage_path, output_dir, args.compression),
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
    parser = argparse.ArgumentParser(description="Convert real MiniCPM5 Core ML graphs.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--graph", choices=["prefill", "prefill-kv", "decode"], default="prefill")
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="artifacts/coreml/real-minicpm5-prefill-16")
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--compression", choices=["none", "int8", "int4"], default=None)
    parser.add_argument("--quantize", action="store_true", help="Deprecated alias for --compression int8.")
    args = parser.parse_args()
    if args.compression is None:
        args.compression = "int8" if args.quantize else "none"
    elif args.quantize and args.compression != "int8":
        parser.error("--quantize can only be combined with --compression int8")
    return args


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
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, ...]:
    input_ids, token_mask, position_ids, causal_mask = build_prefill_tensors(
        tokenizer,
        args.context_tokens,
        args.prompt,
    )
    if args.graph in {"prefill", "prefill-kv"}:
        return input_ids, position_ids, causal_mask
    return build_decode_inputs(model, input_ids, token_mask, position_ids, causal_mask)


def build_prefill_tensors(
    tokenizer,
    context_tokens: int,
    prompt: str = DEFAULT_PROMPT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return input_ids, token_mask, position_ids, causal_mask


def build_4d_causal_mask(token_mask: torch.Tensor) -> torch.Tensor:
    context_tokens = token_mask.shape[-1]
    key_is_real = token_mask[:, None, None, :].to(torch.bool)
    causal = torch.tril(torch.ones((context_tokens, context_tokens), dtype=torch.bool))
    allowed = causal[None, None, :, :] & key_is_real
    blocked = torch.full((1, 1, context_tokens, context_tokens), torch.finfo(torch.float16).min)
    return torch.where(allowed, torch.zeros_like(blocked), blocked).to(torch.float16)


def build_decode_inputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    position_ids: torch.Tensor,
    causal_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        output = model(
            input_ids=input_ids.to(torch.long),
            attention_mask=causal_mask,
            position_ids=position_ids.to(torch.long),
            use_cache=True,
        )

    last_logits = output.logits[:, -1, :]
    token_id = torch.argmax(last_logits, dim=-1, keepdim=True).to(torch.int32)
    real_token_count = token_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    decode_position_id = real_token_count.to(torch.int32)
    decode_mask = build_decode_causal_mask(token_mask)
    cache_tensors: list[torch.Tensor] = []
    for layer in output.past_key_values.layers:
        cache_tensors.append(layer.keys.detach())
        cache_tensors.append(layer.values.detach())
    return tuple([token_id, decode_position_id, decode_mask, *cache_tensors])


def build_decode_causal_mask(token_mask: torch.Tensor) -> torch.Tensor:
    past_tokens = token_mask.shape[-1]
    key_is_real = token_mask[:, None, None, :].to(torch.bool)
    current_token = torch.ones((token_mask.shape[0], 1, 1, 1), dtype=torch.bool)
    allowed = torch.cat([key_is_real, current_token], dim=-1)
    blocked = torch.full((token_mask.shape[0], 1, 1, past_tokens + 1), torch.finfo(torch.float16).min)
    return torch.where(allowed, torch.zeros_like(blocked), blocked).to(torch.float16)


def trace_graph(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> torch.jit.ScriptModule:
    if args.graph == "prefill":
        wrapper: torch.nn.Module = MiniCPMPrefillWrapper(model)
    elif args.graph == "prefill-kv":
        wrapper = MiniCPMPrefillKVWrapper(model)
    else:
        wrapper = MiniCPMDecodeKVWrapper(model, model.config.num_hidden_layers)
    wrapper.eval()
    with torch.no_grad():
        _ = wrapper(*example_inputs)
        return torch.jit.trace(wrapper, example_inputs, strict=False)


def convert_graph(
    traced: torch.jit.ScriptModule,
    example_inputs: tuple[torch.Tensor, ...],
    output_dir: Path,
    num_hidden_layers: int,
    args: argparse.Namespace,
) -> Path:
    package_path = output_dir / f"{args.graph}-{args.context_tokens}.mlpackage"
    if package_path.exists():
        shutil.rmtree(package_path)

    precision = ct.precision.FLOAT16 if args.compute_precision == "float16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.watchOS10,
        compute_precision=precision,
        inputs=input_types(args.graph, example_inputs, num_hidden_layers),
        outputs=output_types(args.graph, num_hidden_layers),
    )
    mlmodel.save(package_path)
    return package_path


def input_types(
    graph: str,
    example_inputs: tuple[torch.Tensor, ...],
    num_hidden_layers: int,
) -> list[ct.TensorType]:
    if graph in {"prefill", "prefill-kv"}:
        input_ids, position_ids, causal_mask = example_inputs
        return [
            ct.TensorType(name="input_ids", shape=tuple(input_ids.shape), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=tuple(position_ids.shape), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=tuple(causal_mask.shape), dtype=np.float16),
        ]

    token_id, position_id, causal_mask, *past_key_values = example_inputs
    types = [
        ct.TensorType(name="token_id", shape=tuple(token_id.shape), dtype=np.int32),
        ct.TensorType(name="position_id", shape=tuple(position_id.shape), dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=tuple(causal_mask.shape), dtype=np.float16),
    ]
    for index in range(num_hidden_layers):
        key = past_key_values[index * 2]
        value = past_key_values[index * 2 + 1]
        types.append(ct.TensorType(name=f"past_key_{index}", shape=tuple(key.shape), dtype=np.float16))
        types.append(ct.TensorType(name=f"past_value_{index}", shape=tuple(value.shape), dtype=np.float16))
    return types


def output_types(graph: str, num_hidden_layers: int) -> list[ct.TensorType]:
    if graph == "prefill":
        return [ct.TensorType(name="logits")]

    outputs = [ct.TensorType(name="logits")]
    prefix = "new" if graph == "decode" else "present"
    for index in range(num_hidden_layers):
        outputs.append(ct.TensorType(name=f"{prefix}_key_{index}"))
        outputs.append(ct.TensorType(name=f"{prefix}_value_{index}"))
    return outputs


def compress_coreml_package(mlpackage_path: Path, output_dir: Path, compression: str) -> Path:
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
        palettize_weights,
    )

    compressed_path = output_dir / f"{mlpackage_path.stem}-{compression}.mlpackage"
    if compressed_path.exists():
        shutil.rmtree(compressed_path)

    model = ct.models.MLModel(str(mlpackage_path))
    if compression == "int8":
        config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric"))
        compressed = linear_quantize_weights(model, config=config)
    elif compression == "int4":
        config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=4))
        compressed = palettize_weights(model, config=config)
    else:
        raise ValueError(f"Unsupported compression: {compression}")

    compressed.save(str(compressed_path))
    return compressed_path


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
