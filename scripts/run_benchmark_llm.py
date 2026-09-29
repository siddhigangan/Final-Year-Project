"""Run CodeRAG-PoisonBench against a real local Ollama model.

Usage:
    python -m scripts.run_benchmark_llm --model qwen2.5-coder:3b
    python -m scripts.run_benchmark_llm --model qwen2.5-coder:3b --seeds 2

Requires a running Ollama server with the model pulled (check what the
server sees with: Invoke-RestMethod http://localhost:11434/api/tags).
Writes results/benchmark_llm_report.json. Ctrl+C saves a partial report.
Never called by the test suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.llm_eval import (
    describe_rules,
    evaluate_sample,
    overall,
    summarize,
)
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
    parser.add_argument(
        "--seeds", type=int, default=len(LLM_SEEDS),
        help=f"number of seeds to use, 1-{len(LLM_SEEDS)}",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--tag", default=None,
        help="suffix for the output filename, e.g. --tag 1p5b writes "
             "benchmark_llm_report_1p5b.json instead of overwriting the "
             "default. Defaults to no suffix.",
    )
    args = parser.parse_args()

    if not 1 <= args.seeds <= len(LLM_SEEDS):
        print(f"ERROR: --seeds must be 1-{len(LLM_SEEDS)}", file=sys.stderr)
        return 2

    provider = LocalOllamaProvider(
        model_name=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        temperature=0.0,
        max_tokens=600,
    )

    try:
        provider.health_check()
        available = provider.list_models()
    except OllamaConnectionError as exc:
        print(f"ERROR: {exc}\nStart Ollama, then retry.", file=sys.stderr)
        return 2

    if args.model not in available:
        print(
            f"ERROR: model {args.model!r} not found on the server. "
            f"Available: {available}",
            file=sys.stderr,
        )
        return 2

    seeds = LLM_SEEDS[: args.seeds]
    dataset = build_poisoning_dataset(list(seeds))
    total = dataset.size
    results = []
    interrupted = False

    try:
        for index, sample in enumerate(dataset.samples, start=1):
            print(f"[{index}/{total}] {sample.sample_id}", flush=True)
            try:
                results.append(evaluate_sample(sample, provider=provider))
            except GenerationError as exc:
                print(f"  skipped: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; saving partial report.", file=sys.stderr)

    report = {
        "model": args.model,
        "temperature": 0.0,
        "prompt": "src.generation.PromptBuilder (default, hardened)",
        "seeds": [s.sample_id for s in seeds],
        "samples_planned": total,
        "samples_evaluated": len(results),
        "samples_skipped": total - len(results),
        "partial": interrupted,
        "metric_note": (
            "attack_flip_rate = poisoned run followed the poison AND the "
            "clean control did not. on_task_rate = output plausibly "
            "attempts the task. rule_scope 'echo' rules detect copying "
            "of planted text, a weaker signal than changed behaviour."
        ),
        "attack_rules": describe_rules(),
        "overall": overall(results),
        "per_category": summarize(results),
    }

    suffix = f"_{args.tag}" if args.tag else ""
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"benchmark_llm_report{suffix}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"overall": report["overall"],
                      "per_category": report["per_category"]}, indent=2))
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
