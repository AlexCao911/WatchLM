#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONVERSION_SCRIPT = ROOT / "tools" / "conversion" / "convert-minicpm5-coreml.py"
DEFAULT_CALIBRATION_PROMPTS = ROOT / "tools" / "benchmark" / "fixtures" / "calibration-prompts.json"
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "hf" / "MiniCPM5-1B"
SOURCE_MODEL_ID = "openbmb/MiniCPM5-1B"
TARGET_COMPONENTS = ["attentionQKO", "attentionV", "ffnGateUp", "ffnDown", "ffn", "embedding", "lmHead", "norms"]
STATISTIC = "sum_input_activation_squared_by_column"


def main() -> None:
    args = parse_args()
    suite = load_calibration_suite(Path(args.calibration_prompts))
    if args.dry_run:
        report = build_report(args, suite, modules=[], elapsed_seconds=0.0, mode="dry-run")
    else:
        started = time.perf_counter()
        modules = collect_importance(args, suite)
        report = build_report(
            args,
            suite,
            modules=modules,
            elapsed_seconds=time.perf_counter() - started,
            mode="activation-collection",
        )

    encoded = json.dumps(report, indent=2, sort_keys=True)
    if not args.quiet:
        print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MiniCPM activation-importance statistics for WatchLM quantization policy search."
    )
    parser.add_argument("--calibration-prompts", default=str(DEFAULT_CALIBRATION_PROMPTS))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--top-columns", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--top-groups", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("--group-size must be positive")
    if args.top_groups <= 0:
        parser.error("--top-groups must be positive")
    return args


def load_calibration_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf8"))
    errors = calibration_suite_errors(suite)
    if errors:
        raise ValueError("Invalid calibration prompt suite:\n- " + "\n- ".join(errors))
    suite["_path"] = str(path)
    return suite


def calibration_suite_errors(suite: Any) -> list[str]:
    if not isinstance(suite, dict):
        return ["suite must be an object"]

    errors: list[str] = []
    if suite.get("schemaVersion") != 1:
        errors.append("schemaVersion must be 1")
    if suite.get("modelId") != SOURCE_MODEL_ID:
        errors.append(f"modelId must be {SOURCE_MODEL_ID}")
    if suite.get("tokenizerSource") != SOURCE_MODEL_ID:
        errors.append(f"tokenizerSource must be {SOURCE_MODEL_ID}")
    if suite.get("contextTokens") != 256:
        errors.append("contextTokens must be 256")
    if suite.get("promptFormat") != "minicpm5-chat-template-no-think":
        errors.append("promptFormat must be minicpm5-chat-template-no-think")
    if not isinstance(suite.get("prefixTokenCounts"), list) or not suite["prefixTokenCounts"]:
        errors.append("prefixTokenCounts must be a non-empty array")
    if not isinstance(suite.get("prompts"), list) or not suite["prompts"]:
        errors.append("prompts must be a non-empty array")
    return errors


def classify_module_name(name: str) -> str | None:
    normalized = name.replace("_", ".")
    if name == "lm_head" or name.endswith(".lm_head") or "output_projection" in name:
        return "lmHead"
    if "embed_tokens" in name or "tok_embeddings" in name or normalized.endswith(".embedding"):
        return "embedding"
    if "norm" in normalized or "layernorm" in normalized:
        return "norms"
    if any(pattern in name for pattern in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj")):
        return "attentionQKO"
    if "self_attn.v_proj" in name:
        return "attentionV"
    if any(pattern in name for pattern in ("mlp.gate_proj", "mlp.up_proj")):
        return "ffnGateUp"
    if "mlp.down_proj" in name:
        return "ffnDown"
    if "feed_forward" in name or re.search(r"(^|\.)ffn(\.|$)", normalized):
        return "ffn"
    return None


def extract_layer_index(name: str) -> int | None:
    match = re.search(r"(?:^|[._])layers?[._](\d+)(?:[._]|$)", name)
    if match:
        return int(match.group(1))
    return None


def collect_importance(args: argparse.Namespace, suite: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    conversion = load_conversion_module()
    snapshot_path = Path(args.cache_dir).resolve()
    tokenizer = conversion.load_tokenizer(snapshot_path)
    model = conversion.load_model(snapshot_path)
    model.eval()
    device = resolve_device(args.device, torch)
    model.to(device)

    stats: dict[str, dict[str, Any]] = {}
    hooks = []
    for name, module in model.named_modules():
        component = classify_module_name(name)
        if component is None or component == "embedding":
            continue
        hooks.append(module.register_forward_pre_hook(make_activation_hook(name, component, stats, torch)))

    prompts = suite["prompts"]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    try:
        with torch.no_grad():
            for prompt in prompts:
                encoded = tokenizer(
                    prompt["renderedPrompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(suite["contextTokens"]),
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    return summarize_stats(
        stats,
        top_columns=args.top_columns,
        group_size=args.group_size,
        top_groups=args.top_groups,
    )


def make_activation_hook(name: str, component: str, stats: dict[str, dict[str, Any]], torch):
    def hook(module, inputs) -> None:
        if not inputs:
            return
        tensor = inputs[0]
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            return

        values = tensor.detach().float()
        flattened = values.reshape(-1, values.shape[-1])
        energy = (flattened * flattened).sum(dim=0).cpu()
        entry = stats.setdefault(
            name,
            {
                "name": name,
                "component": component,
                "layerIndex": extract_layer_index(name),
                "observationCount": 0,
                "energy": torch.zeros_like(energy),
            },
        )
        entry["energy"] += energy
        entry["observationCount"] += int(flattened.shape[0])

    return hook


def summarize_stats(
    stats: dict[str, dict[str, Any]],
    top_columns: int,
    group_size: int = 32,
    top_groups: int = 8,
) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for entry in stats.values():
        energy = entry["energy"]
        total = float(energy.sum().item())
        count = min(top_columns, int(energy.numel()))
        top_values, top_indices = energy.topk(count)
        channel_summary = summarize_channel_energy(energy, top_values, total)
        groups = summarize_channel_groups(energy, group_size=group_size, top_groups=top_groups)
        modules.append(
            {
                "name": entry["name"],
                "component": entry["component"],
                "layerIndex": entry["layerIndex"],
                "inputFeatures": int(energy.numel()),
                "observationCount": entry["observationCount"],
                "totalActivationEnergy": total,
                "meanActivationEnergy": total / max(1, int(energy.numel())),
                "channelGroupSize": group_size,
                "channelGroupCount": max(1, (int(energy.numel()) + group_size - 1) // group_size),
                "channelSummary": channel_summary,
                "topColumns": [
                    {"index": int(index), "energy": float(value)}
                    for index, value in zip(top_indices.tolist(), top_values.tolist())
                ],
                "topGroups": groups,
            }
        )
    return sorted(
        modules,
        key=lambda item: (
            item["component"],
            item["layerIndex"] if item["layerIndex"] is not None else -1,
            item["name"],
        ),
    )


def summarize_channel_energy(energy, top_values, total: float) -> dict[str, float]:
    if int(energy.numel()) == 0 or total <= 0:
        return {
            "maxColumnEnergy": 0.0,
            "topColumnEnergyFraction": 0.0,
            "topColumnsEnergyFraction": 0.0,
        }

    max_energy = float(energy.max().item())
    return {
        "maxColumnEnergy": max_energy,
        "topColumnEnergyFraction": max_energy / total,
        "topColumnsEnergyFraction": float(top_values.sum().item()) / total,
    }


def summarize_channel_groups(energy, group_size: int, top_groups: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    feature_count = int(energy.numel())
    for start in range(0, feature_count, group_size):
        end = min(start + group_size, feature_count)
        values = energy[start:end]
        total = float(values.sum().item())
        if int(values.numel()) > 0:
            local_top_value, local_top_index = values.max(dim=0)
            top_column_energy = float(local_top_value.item())
            top_column_index = start + int(local_top_index.item())
        else:
            top_column_energy = 0.0
            top_column_index = start
        groups.append(
            {
                "groupIndex": len(groups),
                "startColumn": start,
                "endColumnExclusive": end,
                "totalActivationEnergy": total,
                "meanActivationEnergy": total / max(1, end - start),
                "topColumnIndex": top_column_index,
                "topColumnEnergy": top_column_energy,
            }
        )

    return sorted(groups, key=lambda item: item["totalActivationEnergy"], reverse=True)[:top_groups]


def build_report(
    args: argparse.Namespace,
    suite: dict[str, Any],
    modules: list[dict[str, Any]],
    elapsed_seconds: float,
    mode: str,
) -> dict[str, Any]:
    prompts = suite["prompts"]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    return {
        "schemaVersion": 1,
        "sourceModelId": SOURCE_MODEL_ID,
        "calibration": {
            "path": suite.get("_path"),
            "promptCount": len(prompts),
            "contextTokens": suite["contextTokens"],
            "promptFormat": suite["promptFormat"],
            "prefixTokenCounts": suite["prefixTokenCounts"],
            "categories": sorted({prompt["category"] for prompt in prompts}),
        },
        "collection": {
            "mode": mode,
            "statistic": STATISTIC,
            "elapsedSeconds": round(elapsed_seconds, 6),
            "topColumns": args.top_columns,
            "groupSize": args.group_size,
            "topGroups": args.top_groups,
        },
        "targetComponents": TARGET_COMPONENTS,
        "componentSummary": component_summary(modules),
        "layerSummary": layer_summary(modules),
        "modules": modules,
    }


def component_summary(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in modules:
        grouped.setdefault(str(item["component"]), []).append(item)

    summary: list[dict[str, Any]] = []
    for component in sorted(grouped):
        values = [float(item["totalActivationEnergy"]) for item in grouped[component]]
        total = sum(values)
        summary.append(
            {
                "component": component,
                "moduleCount": len(values),
                "totalActivationEnergy": total,
                "meanModuleActivationEnergy": total / max(1, len(values)),
                "maxModuleActivationEnergy": max(values) if values else 0.0,
            }
        )
    return summary


def layer_summary(modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in modules:
        layer_index = item.get("layerIndex")
        if layer_index is None:
            continue
        grouped.setdefault(int(layer_index), []).append(item)

    summary: list[dict[str, Any]] = []
    for layer_index in sorted(grouped):
        items = grouped[layer_index]
        component_totals: dict[str, float] = {}
        for item in items:
            component = str(item["component"])
            component_totals[component] = component_totals.get(component, 0.0) + float(item["totalActivationEnergy"])
        summary.append(
            {
                "layerIndex": layer_index,
                "moduleCount": len(items),
                "totalActivationEnergy": sum(component_totals.values()),
                "componentTotals": dict(sorted(component_totals.items())),
            }
        )
    return summary


def load_conversion_module():
    spec = importlib.util.spec_from_file_location("watchlm_minicpm_conversion", CONVERSION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load conversion script: {CONVERSION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(value: str, torch):
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        return torch.device("mps")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
