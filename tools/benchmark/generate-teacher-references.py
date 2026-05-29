#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "openbmb/MiniCPM5-1B"
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "hf" / "MiniCPM5-1B"
DEFAULT_PROMPTS = ROOT / "tools" / "benchmark" / "fixtures" / "benchmark-prompts.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "benchmarks" / "minicpm5-teacher-references.json"


def main() -> None:
    args = parse_args()
    prompts_path = resolve_repo_path(args.prompts)
    output_path = resolve_repo_path(args.output)
    prompts = load_prompts(prompts_path, args.prompt_limit)

    if args.mock_token_ids:
        token_ids = parse_token_ids(args.mock_token_ids)
        references = build_mock_references(prompts, token_ids)
        source = "mock-teacher"
    else:
        references = generate_teacher_references(prompts, args)
        source = args.source

    sidecar = build_sidecar(args, prompts_path, prompts, source, references)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"wrote teacher references: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prompt-id keyed PyTorch teacher token references for Swift benchmarks."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--source",
        default="pytorch-teacher-minicpm5",
        help="Reference source recorded in the Swift benchmark sidecar.",
    )
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Optional cap applied to each prompt's maxNewTokens for smoke generation.",
    )
    parser.add_argument(
        "--mock-token-ids",
        help="Comma-separated token IDs used for every prompt; avoids importing PyTorch/Transformers.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Teacher generation device. CPU is the most portable baseline.",
    )
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def load_prompts(path: Path, prompt_limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_prompts = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(raw_prompts, list) or not raw_prompts:
        raise ValueError("prompt suite must contain a non-empty prompts array")

    prompts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, prompt in enumerate(raw_prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"prompt[{index}] must be an object")
        prompt_id = str(prompt.get("id", "")).strip()
        text = str(prompt.get("input", "")).strip()
        max_new_tokens = prompt.get("maxNewTokens")
        if not prompt_id:
            raise ValueError(f"prompt[{index}].id must be non-empty")
        if prompt_id in seen_ids:
            raise ValueError(f"prompt[{index}].id must be unique")
        if not text:
            raise ValueError(f"prompt[{index}].input must be non-empty")
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError(f"prompt[{index}].maxNewTokens must be a positive integer")

        seen_ids.add(prompt_id)
        prompts.append(prompt)

    if prompt_limit is not None:
        if prompt_limit <= 0:
            raise ValueError("--prompt-limit must be positive")
        prompts = prompts[:prompt_limit]
    return prompts


def parse_token_ids(value: str) -> list[int]:
    token_ids: list[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        token_id = int(token)
        if token_id < 0:
            raise ValueError("--mock-token-ids must not contain negative values")
        token_ids.append(token_id)
    if not token_ids:
        raise ValueError("--mock-token-ids must contain at least one token id")
    return token_ids


def build_mock_references(prompts: list[dict[str, Any]], token_ids: list[int]) -> list[dict[str, Any]]:
    return [{"promptID": prompt["id"], "tokenIDs": token_ids} for prompt in prompts]


def generate_teacher_references(prompts: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    torch, AutoModelForCausalLM, AutoTokenizer = import_teacher_dependencies()
    cache_dir = resolve_repo_path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"model cache does not exist: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.config._attn_implementation = "eager"
    model.eval()

    device = torch.device(args.device)
    model.to(device)

    references: list[dict[str, Any]] = []
    for prompt in prompts:
        max_new_tokens = int(prompt["maxNewTokens"])
        if args.max_new_tokens is not None:
            if args.max_new_tokens <= 0:
                raise ValueError("--max-new-tokens must be positive")
            max_new_tokens = min(max_new_tokens, args.max_new_tokens)

        encoded = tokenizer(
            prompt["input"],
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_token_count = int(encoded["input_ids"].shape[-1])

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        token_ids = generated[0, prompt_token_count:].detach().cpu().tolist()
        references.append(
            {
                "promptID": prompt["id"],
                "tokenIDs": [int(token_id) for token_id in token_ids],
            }
        )

    return references


def import_teacher_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "PyTorch teacher generation requires torch and transformers. "
            "Use --mock-token-ids for a dependency-free schema smoke run."
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def build_sidecar(
    args: argparse.Namespace,
    prompts_path: Path,
    prompts: list[dict[str, Any]],
    source: str,
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_ids = [str(prompt["id"]) for prompt in prompts]
    if len(references) != len(prompt_ids):
        raise ValueError("reference count must match selected prompt count")
    if [reference["promptID"] for reference in references] != prompt_ids:
        raise ValueError("references must preserve prompt order")

    return {
        "schemaVersion": 1,
        "source": source,
        "modelId": args.model_id,
        "promptSuitePath": repo_relative_path(prompts_path),
        "promptCount": len(prompts),
        "maxNewTokensCap": args.max_new_tokens,
        "generatedAt": int(time.time()),
        "references": references,
    }


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
