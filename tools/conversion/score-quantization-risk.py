#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMPORTANCE_REPORT = ROOT / "artifacts" / "benchmarks" / "minicpm5-activation-importance-cal12-groups.json"

PRECISION_RISK_MULTIPLIERS = {
    "int8": 0.35,
    "uint8": 0.35,
    "int4": 1.0,
    "uint4": 1.0,
    "int3": 1.4,
    "uint3": 1.4,
    "int2": 2.0,
    "uint2": 2.0,
}


def main() -> None:
    args = parse_args()
    importance_path = Path(args.importance_report).resolve()
    policy_path = Path(args.precision_policy).resolve()
    importance_report = json.loads(importance_path.read_text(encoding="utf8"))
    policy = json.loads(policy_path.read_text(encoding="utf8"))
    report = score_policy(
        importance_report,
        policy,
        max_weighted_risk=args.max_weighted_risk,
        max_layer_weighted_risk=args.max_layer_weighted_risk,
        max_top_column_fraction=args.max_top_column_fraction,
        max_top_group_fraction=args.max_top_group_fraction,
        source_report_path=display_path(importance_path),
        policy_path=display_path(policy_path),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a mixed precision policy using activation-weighted low-bit risk."
    )
    parser.add_argument("--importance-report", default=str(DEFAULT_IMPORTANCE_REPORT))
    parser.add_argument("--precision-policy", required=True)
    parser.add_argument("--max-weighted-risk", type=float, default=1.0)
    parser.add_argument("--max-layer-weighted-risk", type=float, default=1.0)
    parser.add_argument("--max-top-column-fraction", type=float, default=1.0)
    parser.add_argument("--max-top-group-fraction", type=float, default=1.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_weighted_risk <= 0:
        parser.error("--max-weighted-risk must be positive")
    if args.max_layer_weighted_risk <= 0:
        parser.error("--max-layer-weighted-risk must be positive")
    if args.max_top_column_fraction <= 0:
        parser.error("--max-top-column-fraction must be positive")
    if args.max_top_group_fraction <= 0:
        parser.error("--max-top-group-fraction must be positive")
    return args


def score_policy(
    importance_report: dict[str, Any],
    policy: dict[str, Any],
    max_weighted_risk: float,
    max_layer_weighted_risk: float,
    max_top_column_fraction: float,
    max_top_group_fraction: float,
    source_report_path: str | None = None,
    policy_path: str | None = None,
) -> dict[str, Any]:
    modules = importance_report.get("modules")
    if not isinstance(modules, list):
        raise ValueError("importance report must contain modules")

    component_totals = component_activation_totals(modules)
    scored_modules = [
        score_module(module, policy, component_totals)
        for module in modules
    ]
    scored_modules = [module for module in scored_modules if module is not None]
    scored_modules.sort(key=lambda item: (-item["weightedRiskScore"], item["name"]))

    failures: list[str] = []
    rejected_count = 0
    for module in scored_modules:
        rejected = False
        if module["weightedRiskScore"] > max_weighted_risk:
            failures.append(
                f'{module["name"]} weightedRiskScore {module["weightedRiskScore"]:.3f} exceeds {max_weighted_risk:.3f}'
            )
            rejected = True
        if module["topColumnEnergyFraction"] > max_top_column_fraction:
            failures.append(
                f'{module["name"]} topColumnEnergyFraction {module["topColumnEnergyFraction"]:.3f} exceeds {max_top_column_fraction:.3f}'
            )
            rejected = True
        if module["topGroupEnergyFraction"] > max_top_group_fraction:
            failures.append(
                f'{module["name"]} topGroupEnergyFraction {module["topGroupEnergyFraction"]:.3f} exceeds {max_top_group_fraction:.3f}'
            )
            rejected = True
        module["rejected"] = rejected
        if rejected:
            rejected_count += 1

    layer_summary = scored_layer_summary(scored_modules)
    rejected_layer_count = 0
    for layer in layer_summary:
        rejected = False
        if layer["weightedRiskScore"] > max_layer_weighted_risk:
            failures.append(
                f'layer {layer["layerIndex"]} weightedRiskScore {layer["weightedRiskScore"]:.3f} exceeds {max_layer_weighted_risk:.3f}'
            )
            rejected = True
        layer["rejected"] = rejected
        if rejected:
            rejected_layer_count += 1

    max_risk = max((module["weightedRiskScore"] for module in scored_modules), default=0.0)
    mean_risk = sum(module["weightedRiskScore"] for module in scored_modules) / max(1, len(scored_modules))
    return {
        "schemaVersion": 1,
        "sourceReport": source_report_path,
        "policyPath": policy_path,
        "policyId": policy.get("policyId"),
        "summary": {
            "scoredModuleCount": len(scored_modules),
            "rejectedModuleCount": rejected_count,
            "rejectedLayerCount": rejected_layer_count,
            "maxWeightedRiskScore": round(max_risk, 6),
            "meanWeightedRiskScore": round(mean_risk, 6),
        },
        "gate": {
            "ok": not failures,
            "failures": failures,
            "targets": {
                "maxWeightedRisk": max_weighted_risk,
                "maxLayerWeightedRisk": max_layer_weighted_risk,
                "maxTopColumnEnergyFraction": max_top_column_fraction,
                "maxTopGroupEnergyFraction": max_top_group_fraction,
            },
        },
        "layerSummary": layer_summary,
        "modules": scored_modules,
    }


def component_activation_totals(modules: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for module in modules:
        component = module.get("component")
        if component is None:
            continue
        totals[str(component)] = totals.get(str(component), 0.0) + float(module.get("totalActivationEnergy", 0.0))
    return totals


def score_module(
    module: dict[str, Any],
    policy: dict[str, Any],
    component_totals: dict[str, float],
) -> dict[str, Any] | None:
    component = module.get("component")
    if not isinstance(component, str):
        return None
    layer_index = module.get("layerIndex")
    precision = effective_precision(policy, component, layer_index)
    multiplier = PRECISION_RISK_MULTIPLIERS.get(precision)
    if multiplier is None:
        return None

    total_energy = float(module.get("totalActivationEnergy", 0.0))
    component_total = component_totals.get(component, 0.0)
    normalized_energy = total_energy / component_total if component_total > 0 else 0.0
    top_column_fraction = top_column_energy_fraction(module)
    top_group_fraction = top_group_energy_fraction(module)
    weighted_risk = multiplier * normalized_energy * (1.0 + top_column_fraction + top_group_fraction)

    return {
        "name": str(module.get("name", "")),
        "component": component,
        "layerIndex": layer_index,
        "precision": precision,
        "precisionRiskMultiplier": multiplier,
        "componentActivationEnergyFraction": round(normalized_energy, 6),
        "topColumnEnergyFraction": round(top_column_fraction, 6),
        "topGroupEnergyFraction": round(top_group_fraction, 6),
        "weightedRiskScore": round(weighted_risk, 3),
        "rejected": False,
    }


def scored_layer_summary(scored_modules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for module in scored_modules:
        layer_index = module.get("layerIndex")
        if layer_index is None:
            continue
        grouped.setdefault(int(layer_index), []).append(module)

    summary: list[dict[str, Any]] = []
    for layer_index in sorted(grouped):
        items = grouped[layer_index]
        score = sum(float(item["weightedRiskScore"]) for item in items)
        summary.append(
            {
                "layerIndex": layer_index,
                "scoredModuleCount": len(items),
                "weightedRiskScore": round(score, 3),
                "rejected": False,
            }
        )
    return summary


def effective_precision(policy: dict[str, Any], component: str, layer_index: Any) -> str:
    precision = (policy.get("weights") or {}).get(component, "fp16")
    if layer_index is not None:
        overrides = (policy.get("layerOverrides") or {}).get(component) or {}
        precision = overrides.get(str(layer_index), precision)
    return str(precision)


def top_column_energy_fraction(module: dict[str, Any]) -> float:
    summary = module.get("channelSummary") or {}
    return float(summary.get("topColumnEnergyFraction", 0.0))


def top_group_energy_fraction(module: dict[str, Any]) -> float:
    total_energy = float(module.get("totalActivationEnergy", 0.0))
    if total_energy <= 0:
        return 0.0
    top_groups = module.get("topGroups") or []
    if not top_groups:
        return 0.0
    group_energy = max(float(group.get("totalActivationEnergy", 0.0)) for group in top_groups)
    return group_energy / total_energy


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
