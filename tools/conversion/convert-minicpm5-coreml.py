#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_PRECISION_POLICY = ROOT / "tools" / "conversion" / "mixed-precision-policy.json"
SUPPORTED_MIXED_PRECISIONS = {"fp16", "int8", "int4"}
SUPPORTED_INT4_COMPRESSION_METHODS = {"palettization"}
SUPPORTED_INT4_PALETTIZATION_MODES = {"kmeans", "uniform"}
SUPPORTED_INT4_PALETTIZATION_GRANULARITIES = {"per_tensor", "per_grouped_channel"}
REQUIRED_MIXED_POLICY_COMPONENTS = ("embedding", "lmHead", "norms", "attentionQKO", "attentionV", "ffn")
FFN_SUBCOMPONENTS = ("ffnGateUp", "ffnDown")
MIXED_POLICY_COMPONENTS = (
    "embedding",
    "lmHead",
    "norms",
    "attentionQKO",
    "attentionV",
    "ffnGateUp",
    "ffnDown",
    "ffn",
)
TRANSFORMER_COMPONENTS = ("attentionQKO", "attentionV", "ffnGateUp", "ffnDown", "ffn")
STATEFUL_GRAPHS = {"stateful-kv", "stateful-step-kv"}
DEFAULT_INT4_COMPRESSION = {
    "method": "palettization",
    "mode": "kmeans",
    "granularity": "per_tensor",
    "groupSize": 32,
    "enablePerChannelScale": False,
    "clusterDim": 1,
    "numKMeansWorkers": 1,
    "weightThreshold": 2048,
}
DEFAULT_OP_NAME_PATTERNS: dict[str, list[str]] = {
    "embedding": ["embed_tokens", "tok_embeddings", "embedding"],
    "lmHead": ["lm_head", "output_projection"],
    "norms": ["input_layernorm", "post_attention_layernorm", "norm"],
    "attentionQKO": [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "attention.wq",
        "attention.wk",
        "attention.wo",
    ],
    "attentionV": ["self_attn.v_proj", "attention.wv"],
    "ffnGateUp": ["mlp.gate_proj", "mlp.up_proj"],
    "ffnDown": ["mlp.down_proj"],
    "ffn": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "feed_forward", "ffn"],
}


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


class MiniCPMStatefulKVWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, context_tokens: int) -> None:
        super().__init__()
        self.model = model
        self.context_tokens = context_tokens
        layer_count = int(model.config.num_hidden_layers)
        kv_heads = config_kv_heads(model.config)
        head_dimension = config_head_dimension(model.config)
        state_shape = (1, kv_heads, context_tokens, head_dimension)
        for index in range(layer_count):
            self.register_buffer(f"past_key_{index}", torch.zeros(state_shape, dtype=torch.float16))
            self.register_buffer(f"past_value_{index}", torch.zeros(state_shape, dtype=torch.float16))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        query_length = input_ids.shape[-1]
        end_step = causal_mask.shape[-1]
        past_kv_length = end_step - query_length
        pairs = []
        for index in range(int(self.model.config.num_hidden_layers)):
            key = getattr(self, f"past_key_{index}")
            value = getattr(self, f"past_value_{index}")
            pairs.append((key[:, :, :past_kv_length, :], value[:, :, :past_kv_length, :]))

        cache = DynamicCache(pairs, config=self.model.config)
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=causal_mask,
            position_ids=position_ids.to(torch.long),
            past_key_values=cache,
            use_cache=True,
        )

        for index, layer in enumerate(output.past_key_values.layers):
            key = getattr(self, f"past_key_{index}")
            value = getattr(self, f"past_value_{index}")
            key[:, :, past_kv_length:end_step, :] = layer.keys[:, :, past_kv_length:end_step, :]
            value[:, :, past_kv_length:end_step, :] = layer.values[:, :, past_kv_length:end_step, :]

        return output.logits[:, -1, :]


class MiniCPMStatefulStepKVWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, context_tokens: int) -> None:
        super().__init__()
        self.model = model
        self.context_tokens = context_tokens
        layer_count = int(model.config.num_hidden_layers)
        kv_heads = config_kv_heads(model.config)
        head_dimension = config_head_dimension(model.config)
        state_shape = (1, kv_heads, context_tokens, head_dimension)
        for index in range(layer_count):
            self.register_buffer(f"past_key_{index}", torch.zeros(state_shape, dtype=torch.float16))
            self.register_buffer(f"past_value_{index}", torch.zeros(state_shape, dtype=torch.float16))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        pairs = []
        for index in range(int(self.model.config.num_hidden_layers)):
            pairs.append((getattr(self, f"past_key_{index}"), getattr(self, f"past_value_{index}")))

        cache = DynamicCache(pairs, config=self.model.config)
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=causal_mask,
            position_ids=position_ids.to(torch.long),
            past_key_values=cache,
            use_cache=True,
        )

        for index, layer in enumerate(output.past_key_values.layers):
            key = getattr(self, f"past_key_{index}")
            value = getattr(self, f"past_value_{index}")
            key[:, :, :, :] = torch.cat([key[:, :, 1:, :], layer.keys[:, :, -1:, :]], dim=2)
            value[:, :, :, :] = torch.cat([value[:, :, 1:, :], layer.values[:, :, -1:, :]], dim=2)

        return output.logits[:, -1, :]


def main() -> None:
    args = parse_args()
    mixed_policy = load_mixed_precision_policy(args.precision_policy) if args.compression == "mixed" else None
    if args.describe_compression_policy:
        print(json.dumps(build_mixed_compression_plan(mixed_policy), indent=2))
        return

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "conversion-report.json"
    report: dict[str, Any] = {
        "modelId": args.model_id,
        "contextTokens": reported_context_tokens(args),
        "computePrecision": args.compute_precision,
        "torchDType": args.torch_dtype,
        "graph": args.graph,
        "compression": args.compression,
        "int4Mode": args.int4_mode if args.compression == "int4" else None,
        "sourceMlpackagePath": args.source_mlpackage,
        "legacyQuantizeFlag": args.quantize,
        "prompt": args.prompt,
        "stages": [],
    }
    if mixed_policy is not None:
        report["precisionPolicyPath"] = report_path_string(resolve_repo_path(args.precision_policy))
        report["mixedPrecisionPolicy"] = build_mixed_compression_plan(mixed_policy)

    try:
        if args.source_mlpackage:
            mlpackage_path = resolve_repo_path(args.source_mlpackage)
        else:
            snapshot_path = run_stage(report, "download_snapshot", lambda: download_snapshot(args))
            tokenizer = run_stage(report, "load_tokenizer", lambda: load_tokenizer(snapshot_path))
            model = run_stage(report, "load_model", lambda: load_model(snapshot_path, args.torch_dtype))
            if args.graph == "stateful-kv":
                report["graphSchema"] = stateful_kv_graph_schema(model.config, args.context_tokens)
            elif args.graph == "stateful-step-kv":
                report["graphSchema"] = stateful_step_kv_graph_schema(model.config, args.context_tokens)
            example_inputs = run_stage(
                report,
                "build_example_inputs",
                lambda: build_example_inputs(model, tokenizer, args),
            )
            traced = run_stage(report, f"trace_{args.graph}", lambda: trace_graph(model, example_inputs, args))
            mlpackage_path = run_stage(
                report,
                f"convert_{args.graph}_coreml",
                lambda: convert_graph(traced, example_inputs, output_dir, model.config, args),
            )

        if args.compression != "none":
            def compress_action() -> Path:
                compressed_path, compression_audit = compress_coreml_package(
                    mlpackage_path,
                    output_dir,
                    args.compression,
                    mixed_policy,
                    int4_mode=args.int4_mode,
                )
                if compression_audit is not None:
                    report["compressionAudit"] = compression_audit
                return compressed_path

            mlpackage_path = run_stage(report, f"compress_coreml_weights_{args.compression}", compress_action)

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
    parser.add_argument(
        "--graph",
        choices=["prefill", "prefill-kv", "decode", "stateful-kv", "stateful-step-kv"],
        default="prefill",
    )
    parser.add_argument("--context-tokens", type=int, default=16)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default="artifacts/coreml/real-minicpm5-prefill-16")
    parser.add_argument(
        "--source-mlpackage",
        default=None,
        help="Compress an existing mlpackage and skip PyTorch tracing/conversion.",
    )
    parser.add_argument("--compute-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--torch-dtype", choices=["float16", "float32", "bfloat16", "auto"], default="float16")
    parser.add_argument("--compression", choices=["none", "int8", "int4", "mixed"], default=None)
    parser.add_argument(
        "--int4-mode",
        choices=["kmeans", "uniform"],
        default="kmeans",
        help="Palettization mode for global --compression int4.",
    )
    parser.add_argument(
        "--precision-policy",
        default=str(DEFAULT_PRECISION_POLICY.relative_to(ROOT)),
        help="JSON policy for --compression mixed.",
    )
    parser.add_argument(
        "--describe-compression-policy",
        action="store_true",
        help="Print the mixed precision compression plan and exit without loading the model.",
    )
    parser.add_argument("--quantize", action="store_true", help="Deprecated alias for --compression int8.")
    args = parser.parse_args()
    if args.compression is None:
        args.compression = "int8" if args.quantize else "none"
    elif args.quantize and args.compression != "int8":
        parser.error("--quantize can only be combined with --compression int8")
    if args.describe_compression_policy and args.compression != "mixed":
        parser.error("--describe-compression-policy requires --compression mixed")
    if args.source_mlpackage and args.compression == "none":
        parser.error("--source-mlpackage requires --compression int8, int4, or mixed")
    return args


def reported_context_tokens(args: argparse.Namespace) -> int:
    if args.source_mlpackage:
        inferred = infer_context_tokens_from_path(args.source_mlpackage)
        if inferred is not None:
            return inferred
    return args.context_tokens


def infer_context_tokens_from_path(path: str | Path) -> int | None:
    name = Path(path).name
    match = re.search(r"(?:^|[-_])(?:context)?(\d+)(?:[-_.]|$)", name)
    if match:
        return int(match.group(1))
    return None


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
        allow_patterns=snapshot_allow_patterns(),
    )
    return Path(path)


def snapshot_allow_patterns() -> list[str]:
    return [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]


def load_tokenizer(snapshot_path: Path):
    return AutoTokenizer.from_pretrained(snapshot_path)


def load_model(snapshot_path: Path, torch_dtype: str = "float16") -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_path,
        torch_dtype=resolve_torch_dtype(torch_dtype),
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.config._attn_implementation = "eager"
    model.eval()
    return model


def resolve_torch_dtype(value: str):
    if value == "auto":
        return "auto"
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported torch dtype: {value}")


def build_example_inputs(
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, ...]:
    if args.graph == "stateful-step-kv":
        return stateful_step_example_inputs(args.context_tokens)

    input_ids, token_mask, position_ids, causal_mask = build_prefill_tensors(
        tokenizer,
        args.context_tokens,
        args.prompt,
    )
    if args.graph in {"prefill", "prefill-kv", "stateful-kv"}:
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
    query_is_padding = ~token_mask[:, None, :, None].to(torch.bool)
    causal = torch.tril(torch.ones((context_tokens, context_tokens), dtype=torch.bool))
    pad_query_self = query_is_padding & torch.eye(context_tokens, dtype=torch.bool)[None, None, :, :]
    allowed = (causal[None, None, :, :] & key_is_real) | pad_query_self
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


def stateful_step_example_inputs(context_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.zeros((1, 1), dtype=torch.int32)
    position_ids = torch.zeros((1, 1), dtype=torch.int32)
    causal_mask = torch.full((1, 1, 1, context_tokens + 1), torch.finfo(torch.float16).min)
    causal_mask[:, :, :, -1:] = 0
    return input_ids, position_ids, causal_mask.to(torch.float16)


def trace_graph(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
) -> torch.jit.ScriptModule:
    if args.graph == "prefill":
        wrapper: torch.nn.Module = MiniCPMPrefillWrapper(model)
    elif args.graph == "prefill-kv":
        wrapper = MiniCPMPrefillKVWrapper(model)
    elif args.graph == "stateful-kv":
        wrapper = MiniCPMStatefulKVWrapper(model, args.context_tokens)
    elif args.graph == "stateful-step-kv":
        wrapper = MiniCPMStatefulStepKVWrapper(model, args.context_tokens)
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
    config: Any,
    args: argparse.Namespace,
) -> Path:
    package_path = output_dir / f"{args.graph}-{args.context_tokens}.mlpackage"
    if package_path.exists():
        shutil.rmtree(package_path)

    precision = ct.precision.FLOAT16 if args.compute_precision == "float16" else ct.precision.FLOAT32
    minimum_deployment_target = ct.target.iOS18 if args.graph in STATEFUL_GRAPHS else ct.target.watchOS10
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=precision,
        inputs=input_types(args.graph, example_inputs, config, args.context_tokens),
        outputs=output_types(args.graph, int(config.num_hidden_layers)),
        states=state_types(args.graph, config, args.context_tokens),
    )
    mlmodel.save(package_path)
    return package_path


def input_types(
    graph: str,
    example_inputs: tuple[torch.Tensor, ...],
    config: Any,
    context_tokens: int,
) -> list[ct.TensorType]:
    if graph == "stateful-step-kv":
        return [
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, context_tokens + 1), dtype=np.float16),
        ]

    if graph == "stateful-kv":
        input_ids, position_ids, causal_mask = example_inputs
        query_length = ct.RangeDim(
            lower_bound=1,
            upper_bound=context_tokens,
            default=int(input_ids.shape[-1]),
        )
        key_length = ct.RangeDim(
            lower_bound=1,
            upper_bound=context_tokens + 1,
            default=int(causal_mask.shape[-1]),
        )
        return [
            ct.TensorType(name="input_ids", shape=(1, query_length), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, query_length), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=(1, 1, query_length, key_length), dtype=np.float16),
        ]

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
    num_hidden_layers = int(config.num_hidden_layers)
    for index in range(num_hidden_layers):
        key = past_key_values[index * 2]
        value = past_key_values[index * 2 + 1]
        types.append(ct.TensorType(name=f"past_key_{index}", shape=tuple(key.shape), dtype=coreml_numpy_dtype(key)))
        types.append(ct.TensorType(name=f"past_value_{index}", shape=tuple(value.shape), dtype=coreml_numpy_dtype(value)))
    return types


def coreml_numpy_dtype(tensor: torch.Tensor) -> Any:
    if tensor.dtype == torch.float16:
        return np.float16
    if tensor.dtype == torch.float32:
        return np.float32
    raise ValueError(f"unsupported Core ML tensor dtype: {tensor.dtype}")


def output_types(graph: str, num_hidden_layers: int) -> list[ct.TensorType]:
    if graph in {"prefill", "stateful-kv", "stateful-step-kv"}:
        return [ct.TensorType(name="logits")]

    outputs = [ct.TensorType(name="logits")]
    prefix = "new" if graph == "decode" else "present"
    for index in range(num_hidden_layers):
        outputs.append(ct.TensorType(name=f"{prefix}_key_{index}"))
        outputs.append(ct.TensorType(name=f"{prefix}_value_{index}"))
    return outputs


def state_types(graph: str, config: Any, context_tokens: int) -> list[ct.StateType] | None:
    if graph not in STATEFUL_GRAPHS:
        return None

    return [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=tuple(spec["shape"]),
                dtype=np.float16,
            ),
            name=spec["name"],
        )
        for spec in stateful_kv_state_specs(config, context_tokens)
    ]


def stateful_kv_graph_schema(config: Any, context_tokens: int) -> dict[str, Any]:
    return {
        "interface": "stateful-kv",
        "layerCount": int(config.num_hidden_layers),
        "kvHeads": config_kv_heads(config),
        "headDimension": config_head_dimension(config),
        "contextTokens": int(context_tokens),
        "inputs": ["input_ids", "position_ids", "causal_mask"],
        "outputs": ["logits"],
        "states": stateful_kv_state_specs(config, context_tokens),
    }


def stateful_step_kv_graph_schema(config: Any, context_tokens: int) -> dict[str, Any]:
    return {
        "interface": "stateful-step-kv",
        "layerCount": int(config.num_hidden_layers),
        "kvHeads": config_kv_heads(config),
        "headDimension": config_head_dimension(config),
        "contextTokens": int(context_tokens),
        "inputs": ["input_ids", "position_ids", "causal_mask"],
        "inputShapes": {
            "input_ids": [1, 1],
            "position_ids": [1, 1],
            "causal_mask": [1, 1, 1, int(context_tokens) + 1],
        },
        "outputs": ["logits"],
        "states": stateful_kv_state_specs(config, context_tokens),
        "stateUpdate": "sliding-window-full-state-write",
    }


def stateful_kv_state_specs(config: Any, context_tokens: int) -> list[dict[str, Any]]:
    layer_count = int(config.num_hidden_layers)
    shape = [1, config_kv_heads(config), int(context_tokens), config_head_dimension(config)]
    states: list[dict[str, Any]] = []
    for layer in range(layer_count):
        states.append({"name": f"past_key_{layer}", "shape": shape, "dtype": "float16"})
        states.append({"name": f"past_value_{layer}", "shape": shape, "dtype": "float16"})
    return states


def config_kv_heads(config: Any) -> int:
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is None:
        kv_heads = getattr(config, "num_attention_heads")
    return int(kv_heads)


def config_head_dimension(config: Any) -> int:
    if hasattr(config, "head_dim"):
        return int(config.head_dim)
    return int(config.hidden_size) // int(config.num_attention_heads)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved.resolve()


def report_path_string(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_mixed_precision_policy(policy_path: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(policy_path)
    policy = json.loads(path.read_text())
    quantization = policy.get("quantization", {})

    weights = policy.get("weights") or quantization.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("mixed precision policy must include weights")

    normalized_weights: dict[str, str] = {}
    for component in REQUIRED_MIXED_POLICY_COMPONENTS:
        precision = weights.get(component)
        if precision not in SUPPORTED_MIXED_PRECISIONS:
            raise ValueError(f"mixed precision policy weights.{component} must be fp16, int8, or int4")
        normalized_weights[component] = precision
    for component in FFN_SUBCOMPONENTS:
        precision = weights.get(component, normalized_weights["ffn"])
        if precision not in SUPPORTED_MIXED_PRECISIONS:
            raise ValueError(f"mixed precision policy weights.{component} must be fp16, int8, or int4")
        normalized_weights[component] = precision

    strategy = policy.get("strategy") or quantization.get("strategy")
    if strategy != "mixed-precision-fidelity-first":
        raise ValueError("mixed precision policy strategy must be mixed-precision-fidelity-first")

    kv_cache = policy.get("kvCache") or quantization.get("kvCache")
    if kv_cache not in {"fp16", "int8"}:
        raise ValueError("mixed precision policy kvCache must be fp16 or int8")

    structural_reduction = policy.get("structuralReduction", quantization.get("structuralReduction"))
    if structural_reduction is not False:
        raise ValueError("mixed precision policy structuralReduction must be false")

    layer_count = int(policy.get("layerCount") or policy.get("architecture", {}).get("layers") or 24)
    if layer_count <= 0:
        raise ValueError("mixed precision policy layerCount must be positive")

    protected_edge_layer_count = int(policy.get("protectedEdgeLayerCount", 2))
    if protected_edge_layer_count < 0:
        raise ValueError("mixed precision policy protectedEdgeLayerCount must be non-negative")

    op_name_patterns = {
        component: list(DEFAULT_OP_NAME_PATTERNS[component])
        for component in MIXED_POLICY_COMPONENTS
    }
    for component, patterns in (policy.get("opNamePatterns") or {}).items():
        if component not in op_name_patterns:
            raise ValueError(f"unsupported mixed precision opNamePatterns component: {component}")
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) and pattern for pattern in patterns):
            raise ValueError(f"mixed precision policy opNamePatterns.{component} must be a non-empty string array")
        op_name_patterns[component] = patterns

    layer_overrides = parse_layer_overrides(policy.get("layerOverrides") or {}, layer_count)
    int4_compression = parse_int4_compression(policy.get("int4Compression"))

    return {
        "schemaVersion": policy.get("schemaVersion", 1),
        "policyId": policy.get("policyId") or policy.get("quantizationPolicyId") or "mixed-int4-ffn-int8-attn-kv",
        "strategy": strategy,
        "layerCount": layer_count,
        "protectedEdgeLayerCount": min(protected_edge_layer_count, layer_count),
        "weights": normalized_weights,
        "kvCache": kv_cache,
        "structuralReduction": structural_reduction,
        "opNamePatterns": op_name_patterns,
        "layerOverrides": layer_overrides,
        "int4Compression": int4_compression,
        "ffnSubcomponentMode": any(
            component in weights
            or component in (policy.get("opNamePatterns") or {})
            or component in layer_overrides
            for component in FFN_SUBCOMPONENTS
        ),
    }


def parse_layer_overrides(raw_overrides: Any, layer_count: int) -> dict[str, dict[int, str]]:
    if raw_overrides == {}:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("mixed precision policy layerOverrides must be an object")

    overrides: dict[str, dict[int, str]] = {}
    for component, component_overrides in raw_overrides.items():
        if component not in TRANSFORMER_COMPONENTS:
            raise ValueError(f"mixed precision policy layerOverrides.{component} is not supported")
        if not isinstance(component_overrides, dict):
            raise ValueError(f"mixed precision policy layerOverrides.{component} must be an object")

        parsed_component: dict[int, str] = {}
        for raw_layer, raw_precision in component_overrides.items():
            try:
                layer = int(raw_layer)
            except ValueError as error:
                raise ValueError(f"mixed precision policy layerOverrides.{component} layer must be an integer") from error
            if layer < 0 or layer >= layer_count:
                raise ValueError(f"mixed precision policy layerOverrides.{component}.{layer} is outside layer count")
            if raw_precision not in SUPPORTED_MIXED_PRECISIONS:
                raise ValueError(
                    f"mixed precision policy layerOverrides.{component}.{layer} must be fp16, int8, or int4"
                )
            parsed_component[layer] = raw_precision
        overrides[component] = parsed_component
    return overrides


def parse_int4_compression(raw_compression: Any) -> dict[str, Any]:
    if raw_compression is None:
        return dict(DEFAULT_INT4_COMPRESSION)
    if not isinstance(raw_compression, dict):
        raise ValueError("mixed precision policy int4Compression must be an object")

    compression = dict(DEFAULT_INT4_COMPRESSION)
    compression.update(raw_compression)

    method = compression.get("method")
    if method not in SUPPORTED_INT4_COMPRESSION_METHODS:
        raise ValueError("mixed precision policy int4Compression.method must be palettization")

    mode = compression.get("mode")
    if mode not in SUPPORTED_INT4_PALETTIZATION_MODES:
        raise ValueError("mixed precision policy int4Compression.mode must be kmeans or uniform")

    granularity = compression.get("granularity")
    if granularity not in SUPPORTED_INT4_PALETTIZATION_GRANULARITIES:
        raise ValueError(
            "mixed precision policy int4Compression.granularity must be per_tensor or per_grouped_channel"
        )

    group_size = int(compression.get("groupSize"))
    if group_size <= 0:
        raise ValueError("mixed precision policy int4Compression.groupSize must be positive")
    compression["groupSize"] = group_size

    cluster_dim = int(compression.get("clusterDim"))
    if cluster_dim <= 0:
        raise ValueError("mixed precision policy int4Compression.clusterDim must be positive")
    compression["clusterDim"] = cluster_dim

    num_kmeans_workers = int(compression.get("numKMeansWorkers"))
    if num_kmeans_workers <= 0:
        raise ValueError("mixed precision policy int4Compression.numKMeansWorkers must be positive")
    compression["numKMeansWorkers"] = num_kmeans_workers

    weight_threshold = compression.get("weightThreshold")
    if weight_threshold is not None:
        weight_threshold = int(weight_threshold)
        if weight_threshold < 0:
            raise ValueError("mixed precision policy int4Compression.weightThreshold must be non-negative")
    compression["weightThreshold"] = weight_threshold

    enable_per_channel_scale = compression.get("enablePerChannelScale")
    if not isinstance(enable_per_channel_scale, bool):
        raise ValueError("mixed precision policy int4Compression.enablePerChannelScale must be boolean")

    return compression


def build_mixed_compression_plan(policy: dict[str, Any]) -> dict[str, Any]:
    layer_precision = {
        str(layer): {
            component: precision_for_component(policy, component, layer)
            for component in TRANSFORMER_COMPONENTS
        }
        for layer in range(policy["layerCount"])
    }
    compression_passes = [
        {
            "precision": "int8",
            "method": "linear_symmetric",
            "opNamePatterns": op_patterns_for_precision(policy, "int8", layer_precision),
        },
        {
            "precision": "int4",
            "method": "kmeans_palettization",
            "opNamePatterns": op_patterns_for_precision(policy, "int4", layer_precision),
            "settings": policy["int4Compression"],
        },
    ]
    compression_passes = [pass_ for pass_ in compression_passes if pass_["opNamePatterns"]]

    return {
        "policyId": policy["policyId"],
        "strategy": policy["strategy"],
        "layerCount": policy["layerCount"],
        "protectedEdgeLayerCount": policy["protectedEdgeLayerCount"],
        "componentPrecision": policy["weights"],
        "layerOverrides": {
            component: {str(layer): precision for layer, precision in sorted(overrides.items())}
            for component, overrides in policy.get("layerOverrides", {}).items()
        },
        "layerPrecision": layer_precision,
        "kvCachePrecision": policy["kvCache"],
        "structuralReduction": policy["structuralReduction"],
        "compressionPasses": compression_passes,
    }


def new_mixed_compression_audit(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "policyId": policy["policyId"],
        "passes": {
            "int8": new_mixed_compression_pass_audit(),
            "int4": new_mixed_compression_pass_audit(),
        },
    }


def new_mixed_compression_pass_audit() -> dict[str, Any]:
    return {
        "selectedOpCount": 0,
        "rejectedOpCount": 0,
        "selectedByComponent": {},
        "selectedByLayer": {},
        "selectedSampleOpNames": [],
    }


def record_mixed_compression_audit(
    audit: dict[str, Any] | None,
    target_precision: str,
    op_name: str,
    component: str | None,
    layer: int | None,
    selected: bool,
) -> None:
    if audit is None:
        return

    pass_audit = audit["passes"][target_precision]
    if not selected:
        pass_audit["rejectedOpCount"] += 1
        return

    pass_audit["selectedOpCount"] += 1
    if component is not None:
        selected_by_component = pass_audit["selectedByComponent"]
        selected_by_component[component] = selected_by_component.get(component, 0) + 1
    if layer is not None:
        selected_by_layer = pass_audit["selectedByLayer"]
        layer_key = str(layer)
        selected_by_layer[layer_key] = selected_by_layer.get(layer_key, 0) + 1
    if len(pass_audit["selectedSampleOpNames"]) < 20:
        pass_audit["selectedSampleOpNames"].append(op_name)


def op_patterns_for_precision(
    policy: dict[str, Any],
    precision: str,
    layer_precision: dict[str, dict[str, str]],
) -> list[str]:
    patterns: list[str] = []
    for component in MIXED_POLICY_COMPONENTS:
        component_precision = policy["weights"][component]
        layer_uses_precision = any(
            layer.get(component) == precision
            for layer in layer_precision.values()
        )
        if component_precision == precision or layer_uses_precision:
            patterns.extend(policy["opNamePatterns"][component])
    return sorted(set(patterns))


def precision_for_component(policy: dict[str, Any], component: str, layer: int | None = None) -> str:
    precision = policy["weights"][component]
    if layer is None or component not in TRANSFORMER_COMPONENTS:
        return precision

    precision = policy.get("layerOverrides", {}).get(component, {}).get(layer, precision)
    if is_protected_layer(policy, layer):
        return raise_precision_to_int8(precision)
    return precision


def is_protected_layer(policy: dict[str, Any], layer: int) -> bool:
    protected_count = policy["protectedEdgeLayerCount"]
    layer_count = policy["layerCount"]
    return layer < protected_count or layer >= layer_count - protected_count


def raise_precision_to_int8(precision: str) -> str:
    return "int8" if precision == "int4" else precision


def compress_coreml_package(
    mlpackage_path: Path,
    output_dir: Path,
    compression: str,
    mixed_policy: dict[str, Any] | None = None,
    int4_mode: str = "kmeans",
) -> tuple[Path, dict[str, Any] | None]:
    from coremltools.optimize.coreml import (
        get_weights_metadata,
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
        compression_audit = None
    elif compression == "int4":
        config = OptimizationConfig(
            global_config=OpPalettizerConfig(**global_int4_palettizer_settings(int4_mode))
        )
        compressed = palettize_weights(model, config=config)
        compression_audit = None
    elif compression == "mixed":
        if mixed_policy is None:
            raise ValueError("mixed compression requires a precision policy")
        compression_audit = new_mixed_compression_audit(mixed_policy)
        int8_op_configs = mixed_precision_op_name_configs(
            model,
            get_weights_metadata,
            mixed_policy,
            "int8",
            OpLinearQuantizerConfig(mode="linear_symmetric"),
            compression_audit,
        )
        if int8_op_configs:
            compressed = linear_quantize_weights(model, config=OptimizationConfig(op_name_configs=int8_op_configs))
        else:
            compressed = model
        int4_settings = mixed_policy["int4Compression"]
        int4_op_configs = mixed_precision_op_name_configs(
            compressed,
            get_weights_metadata,
            mixed_policy,
            "int4",
            OpPalettizerConfig(
                mode=int4_settings["mode"],
                nbits=4,
                granularity=int4_settings["granularity"],
                group_size=int4_settings["groupSize"],
                enable_per_channel_scale=int4_settings["enablePerChannelScale"],
                cluster_dim=int4_settings["clusterDim"],
                num_kmeans_workers=int4_settings["numKMeansWorkers"],
                weight_threshold=int4_settings["weightThreshold"],
            ),
            compression_audit,
        )
        if int4_op_configs:
            compressed = palettize_weights(compressed, config=OptimizationConfig(op_name_configs=int4_op_configs))
    else:
        raise ValueError(f"Unsupported compression: {compression}")

    compressed.save(str(compressed_path))
    return compressed_path, compression_audit


def global_int4_palettizer_settings(args: Any) -> dict[str, Any]:
    mode = getattr(args, "int4_mode", args)
    if mode not in SUPPORTED_INT4_PALETTIZATION_MODES:
        raise ValueError("global int4 palettization mode must be kmeans or uniform")
    return {
        "mode": mode,
        "nbits": 4,
    }


def mixed_precision_op_name_configs(
    model: Any,
    get_weights_metadata: Any,
    policy: dict[str, Any],
    target_precision: str,
    config: Any,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = get_weights_metadata(model, weight_threshold=0)
    op_name_configs: dict[str, Any | None] = {}
    for name in sorted(metadata.keys()):
        component = classify_component_from_op_name(
            name,
            policy["opNamePatterns"],
            ffn_subcomponent_mode=policy.get("ffnSubcomponentMode", False),
        )
        layer = extract_layer_index(name)
        selected = (
            component is not None
            and precision_for_component(policy, component, layer) == target_precision
        )
        record_mixed_compression_audit(audit, target_precision, name, component, layer, selected)
        op_name_configs[name] = config if selected else None
    return op_name_configs


def make_mixed_precision_op_selector(
    policy: dict[str, Any],
    target_precision: str,
    audit: dict[str, Any] | None = None,
):
    def selector(op) -> bool:
        name = getattr(op, "name", "")
        component = classify_component_from_op_name(
            name,
            policy["opNamePatterns"],
            ffn_subcomponent_mode=policy.get("ffnSubcomponentMode", False),
        )
        if component is None:
            record_mixed_compression_audit(audit, target_precision, name, component, None, False)
            return False
        layer = extract_layer_index(name)
        selected = precision_for_component(policy, component, layer) == target_precision
        record_mixed_compression_audit(audit, target_precision, name, component, layer, selected)
        return selected

    return selector


def classify_component_from_op_name(
    name: str,
    op_name_patterns: dict[str, list[str]],
    ffn_subcomponent_mode: bool = False,
) -> str | None:
    for component in MIXED_POLICY_COMPONENTS:
        if component in FFN_SUBCOMPONENTS and not ffn_subcomponent_mode:
            continue
        for pattern in op_name_patterns[component]:
            if pattern_matches_name(pattern, name):
                return component
    return None


def pattern_matches_name(pattern: str, name: str) -> bool:
    lowered_name = name.lower()
    lowered_pattern = pattern.lower()
    if lowered_pattern in lowered_name:
        return True
    return normalize_op_name_token(lowered_pattern) in normalize_op_name_token(lowered_name)


def normalize_op_name_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def extract_layer_index(name: str) -> int | None:
    lowered_name = name.lower()
    for pattern in (
        r"(?:^|[._/\-])layers?[._/\-]?(\d+)(?:[._/\-]|$)",
        r"(?:^|[._/\-])blocks?[._/\-]?(\d+)(?:[._/\-]|$)",
        r"(?:^|[._/\-])h[._/\-]?(\d+)(?:[._/\-]|$)",
    ):
        match = re.search(pattern, lowered_name)
        if match:
            return int(match.group(1))
    return None


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
