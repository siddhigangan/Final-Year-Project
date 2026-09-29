"""Measure the real L1-L5 defense pipeline. No LLM, no network.

Usage: python -m scripts.run_defense_eval
Writes results/defense_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.defense_eval import evaluate_defense
from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.seeds import LLM_SEEDS


def main() -> None:
    dataset = build_poisoning_dataset(list(LLM_SEEDS))
    report = {
        "seeds": [s.sample_id for s in LLM_SEEDS],
        "samples": dataset.size,
        "per_category": evaluate_defense(dataset),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "defense_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
