"""Real LLM, WITH the L1-L5 defense screening context first.

Usage: python -m scripts.run_benchmark_llm_defended --model qwen2.5-coder:3b --seeds 1
Writes results/benchmark_llm_defended_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.benchmark.defense_eval import default_pipeline
from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.llm_eval import evaluate_sample, overall, summarize
from src.benchmark.seeds import LLM_SEEDS
from src.generation.base import GenerationError
from src.generation.local_ollama import (
    LocalOllamaProvider,
    OllamaConnectionError,
)


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
    defense = default_pipeline()
    results = []
    for index, sample in enumerate(dataset.samples, start=1):
        print(f"[{index}/{dataset.size}] {sample.sample_id}", flush=True)
        try:
            results.append(evaluate_sample(sample, provider=provider, defense=defense))
        except GenerationError as exc:
            print(f"  skipped: {exc}", file=sys.stderr)

    report = {
        "model": args.model,
        "defense": "L1-L5 default pipeline, screens context before generation",
        "samples": dataset.size,
        "overall": overall(results),
        "per_category": summarize(results),
        "note": (
            "blocked_rate: sample never reached the model. attack_flip_rate "
            "is computed only over unblocked calls that reached it."
        ),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "benchmark_llm_defended_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"overall": report["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
