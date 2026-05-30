#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMPORTANCE_REPORT = ROOT / "artifacts" / "benchmarks" / "minicpm5-activation-importance-cal12.json"
SUPPORTED_COMPONENTS = {"attentionQKO", "attentionV", "ffn"}


def main() -> None:
    args = parse_args()
    report_path = Path(args.importance_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf8"))
    report["_path"] = display_path(report_path)
    policy = suggest_policy(
        report,
        component=args.component,
        candidate_count=args.candidate_count,
        protected_edge_layer_count=args.protected_edge_layer_count,
        explicit_excluded_layers=parse_layer_list(args.exclude_layers),
        max_top_column_fraction=args.max_top_column_fraction,
        policy_id=args.policy_id,
    )
    encoded = json.dumps(policy, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest a conservative mixed precision policy from activation-importance evidence."
    )
    parser.add_argument("--importance-report", default=str(DEFAULT_IMPORTANCE_REPORT))
    parser.add_argument("--component", choices=sorted(SUPPORTED_COMPONENTS), required=True)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--protected-edge-layer-count", type=int, default=4)
    parser.add_argument("--exclude-layers", default="")
    parser.add_argument("--max-top-column-fraction", type=float)
    parser.add_argument("--policy-id")
    parser.add_argument("--output")
    return parser.parse_args()


def suggest_policy(
    report: dict[str, Any],
    component: str,
    candidate_count: int,
    protected_edge_layer_count: int,
    explicit_excluded_layers: set[int],
    max_top_column_fraction: float | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    if component not in SUPPORTED_COMPONENTS:
        raise ValueError(f"unsupported component: {component}")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if protected_edge_layer_count < 0:
        raise ValueError("protected_edge_layer_count must be non-negative")

    layers = sorted(report["layerSummary"], key=lambda item: int(item["layerIndex"]))
    layer_count = (max(int(item["layerIndex"]) for item in layers) + 1) if layers else 24
    static_excluded_layers = protected_layers(layer_count, protected_edge_layer_count) | explicit_excluded_layers
    layer_risk = component_layer_risk(report.get("modules", []), component)
    risk_excluded_layers = risk_excluded(layer_risk, max_top_column_fraction)
    excluded_layers = static_excluded_layers | risk_excluded_layers
    candidates = candidate_layers(layers, component, candidate_count, excluded_layers, layer_risk)

    selected_overrides = {str(item["layerIndex"]): "int4" for item in candidates}
    resolved_policy_id = policy_id or f"importance-{component}-low{len(candidates)}-int4-rest-fp16"
    ranking = "lowest_component_activation_energy"
    if max_top_column_fraction is not None:
        ranking = "lowest_component_activation_energy_with_channel_risk_filter"
    candidate_evidence: dict[str, Any] = {
        "sourceReport": report.get("_path"),
        "sourcePromptCount": report.get("calibration", {}).get("promptCount"),
        "component": component,
        "ranking": ranking,
        "excludedLayers": sorted(excluded_layers),
        "selectedLayers": candidates,
    }
    if max_top_column_fraction is not None:
        candidate_evidence["channelRisk"] = {
            "maxTopColumnEnergyFraction": max_top_column_fraction,
        }
    return {
        "schemaVersion": 1,
        "policyId": resolved_policy_id,
        "strategy": "mixed-precision-fidelity-first",
        "layerCount": layer_count,
        "protectedEdgeLayerCount": 0,
        "weights": protected_weights(),
        "layerOverrides": {
            component: selected_overrides,
        },
        "kvCache": "fp16",
        "structuralReduction": False,
        "candidateEvidence": candidate_evidence,
    }


def candidate_layers(
    layer_summary: list[dict[str, Any]],
    component: str,
    candidate_count: int,
    excluded_layers: set[int],
    layer_risk: dict[int, float],
) -> list[dict[str, Any]]:
    ranked = []
    for layer in layer_summary:
        layer_index = int(layer["layerIndex"])
        if layer_index in excluded_layers:
            continue
        energy = float(layer.get("componentTotals", {}).get(component, 0.0))
        if energy <= 0:
            continue
        candidate = {
            "layerIndex": layer_index,
            "componentActivationEnergy": energy,
        }
        if layer_index in layer_risk:
            candidate["maxTopColumnEnergyFraction"] = layer_risk[layer_index]
        ranked.append(candidate)
    return sorted(ranked, key=lambda item: (item["componentActivationEnergy"], item["layerIndex"]))[:candidate_count]


def component_layer_risk(modules: list[dict[str, Any]], component: str) -> dict[int, float]:
    risks: dict[int, float] = {}
    for module in modules:
        if module.get("component") != component or module.get("layerIndex") is None:
            continue
        channel_summary = module.get("channelSummary") or {}
        fraction = channel_summary.get("topColumnEnergyFraction")
        if fraction is None:
            continue
        layer_index = int(module["layerIndex"])
        risks[layer_index] = max(risks.get(layer_index, 0.0), float(fraction))
    return risks


def risk_excluded(layer_risk: dict[int, float], max_top_column_fraction: float | None) -> set[int]:
    if max_top_column_fraction is None:
        return set()
    if max_top_column_fraction <= 0:
        raise ValueError("max_top_column_fraction must be positive")
    return {
        layer_index
        for layer_index, fraction in layer_risk.items()
        if fraction > max_top_column_fraction
    }


def protected_layers(layer_count: int, edge_count: int) -> set[int]:
    layers: set[int] = set()
    for index in range(min(edge_count, layer_count)):
        layers.add(index)
        layers.add(layer_count - index - 1)
    return layers


def protected_weights() -> dict[str, str]:
    return {
        "embedding": "fp16",
        "lmHead": "fp16",
        "norms": "fp16",
        "attentionQKO": "fp16",
        "attentionV": "fp16",
        "ffn": "fp16",
    }


def parse_layer_list(raw_value: str) -> set[int]:
    layers: set[int] = set()
    if not raw_value:
        return layers
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        layers.add(int(stripped))
    return layers


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
