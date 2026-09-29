"""Prompt ablation: naive prompt vs hardened prompt, real LLM, no defense.

Usage: python -m scripts.run_ablation --model qwen2.5-coder:3b --seeds 1
Writes results/ablation_report.json. Same guard behavior as
run_benchmark_llm.py: exits cleanly if Ollama or the model is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.llm_eval import evaluate_sample, overall, summarize
from src.benchmark.seeds import LLM_SEEDS
from src.generation.base import GenerationError
from src.generation.local_ollama import (
    LocalOllamaProvider,
    OllamaConnectionError,
)


def run_style(dataset, provider, style: str) -> list:
    results = []
    for index, sample in enumerate(dataset.samples, start=1):
        print(f"  [{style}] [{index}/{dataset.size}] {sample.sample_id}", flush=True)
        try:
            results.append(
                evaluate_sample(sample, provider=provider, prompt_style=style)
            )
        except GenerationError as exc:
            print(f"    skipped: {exc}", file=sys.stderr)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--seeds", type=int, default=1, help=f"1-{len(LLM_SEEDS)}")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    provider = LocalOllamaProvider(
        model_name=args.model, base_url=args.base_url,
        timeout_seconds=args.timeout, temperature=0.0, max_tokens=600,
    )
    try:
        provider.health_check()
        available = provider.list_models()
    except OllamaConnectionError as exc:
        print(f"ERROR: {exc}\nStart Ollama, then retry.", file=sys.stderr)
        return 2
    if args.model not in available:
        print(f"ERROR: model {args.model!r} not found. Available: {available}",
              file=sys.stderr)
        return 2

    dataset = build_poisoning_dataset(list(LLM_SEEDS[: args.seeds]))
    naive = run_style(dataset, provider, "naive")
    hardened = run_style(dataset, provider, "hardened")

    report = {
        "model": args.model,
        "temperature": 0.0,
        "samples_per_style": dataset.size,
        "naive": {"overall": overall(naive), "per_category": summarize(naive)},
        "hardened": {"overall": overall(hardened), "per_category": summarize(hardened)},
        "note": (
            "Same samples, same model, only the prompt differs. A gap "
            "between naive and hardened attack_flip_rate is the effect "
            "of prompt-level defense alone, with no L1-L5 screening."
        ),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "naive_flip_rate": report["naive"]["overall"]["attack_flip_rate"],
        "hardened_flip_rate": report["hardened"]["overall"]["attack_flip_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
