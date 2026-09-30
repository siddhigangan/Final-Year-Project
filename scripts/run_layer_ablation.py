"""Defense layer ablation: what does each L1-L5 layer contribute?

Blueprint section 32. No LLM, no network -- pure defense-pipeline
measurement. Usage: python -m scripts.run_layer_ablation
Writes results/layer_ablation_report.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.benchmark.integration import build_poisoning_dataset
from src.benchmark.layer_ablation import (
    evaluate_layer_ablation,
    evaluate_layer_ablation_by_category,
)
from src.benchmark.seeds import LLM_SEEDS


def main() -> None:
    dataset = build_poisoning_dataset(list(LLM_SEEDS))
    report = {
        "samples": dataset.size,
        "note": (
            "L3 (context validation) never emits a SecurityFinding in "
            "the current pipeline, so its ladder step always matches "
            "the step before it. This is a real property of the "
            "pipeline, not a measurement gap."
        ),
        "overall": evaluate_layer_ablation(dataset),
        "per_category": evaluate_layer_ablation_by_category(dataset),
    }
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "layer_ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
