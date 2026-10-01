"""Utility impact: does the defense flag real, never-poisoned code?

Blueprint section 60, research question 5. No LLM, no network. Reuses
retrieval_eval.collect_distractors to pull real functions out of a real
codebase (default: this repo's own src/) as the test set.

Usage: python -m scripts.run_utility_eval [--root src] [--limit 60]
Writes results/utility_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.retrieval_eval import collect_distractors
from src.benchmark.utility_eval import UTILITY_PROBE_QUERY, evaluate_utility_impact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="src")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    distractors = collect_distractors(Path(args.root), limit=args.limit)
    result = evaluate_utility_impact(distractors)

    report = {
        "source_root": args.root,
        "probe_query": UTILITY_PROBE_QUERY,
        "note": (
            "Measures the defense's false-positive rate against real, "
            "never-poisoned functions, not the 5-snippet poisoning "
            "controls reported elsewhere. Every chunk is screened "
            "against the same labeled placeholder query, not a real "
            "retrieval query, since these functions were not retrieved "
            "for any specific task."
        ),
        **result,
    }

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "utility_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
