"""Retrieval-stage poisoning: does a poisoned chunk get retrieved?

Uses this repository's own src/ as the distractor knowledge base (real
functions, real FAISS index) and a deterministic lexical embedder, so
it needs no model download. Not a semantic-embedding result; see
docs/benchmark.md for what that would take instead.

Usage: python -m scripts.run_retrieval_eval [--top-k 5] [--root src]
Writes results/retrieval_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.retrieval_eval import (
    HashingEmbeddingProvider,
    collect_distractors,
    evaluate_retrieval,
    summarize_retrieval,
)
from src.benchmark.seeds import LLM_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--root", default="src")
    args = parser.parse_args()

    dataset = build_poisoning_dataset(list(LLM_SEEDS))
    distractors = collect_distractors(Path(args.root))
    outcomes = evaluate_retrieval(
        dataset,
        provider=HashingEmbeddingProvider(),
        distractors=distractors,
        top_k=args.top_k,
    )
    report = {
        "embedder": "hashing-bow (lexical, offline, not semantic)",
        "distractor_root": args.root,
        "distractor_count": len(distractors),
        "top_k": args.top_k,
        "samples": dataset.size,
        "per_category": summarize_retrieval(outcomes),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "retrieval_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
