"""Benchmark execution engine for SecureCodeRAG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.benchmark.dataset import BenchmarkDataset, BenchmarkSample
from src.benchmark.metrics import MetricResult, calculate_metrics


class BenchmarkRunnerError(Exception):
    """Base benchmark runner exception."""


@dataclass(frozen=True, slots=True)
class SampleEvaluation:
    """Evaluation result for one benchmark sample."""

    sample_id: str
    clean_success: bool
    poisoned_success: bool
    defended_success: bool
    detected: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise BenchmarkRunnerError(
                "sample_id must be non-empty."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "clean_success": self.clean_success,
            "poisoned_success": self.poisoned_success,
            "defended_success": self.defended_success,
            "detected": self.detected,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Complete benchmark result."""

    evaluations: tuple[SampleEvaluation, ...]
    metrics: MetricResult

    @property
    def sample_count(self) -> int:
        return len(self.evaluations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "metrics": self.metrics.to_dict(),
            "evaluations": [
                evaluation.to_dict()
                for evaluation in self.evaluations
            ],
        }


Evaluator = Callable[
    [BenchmarkSample, str, bool],
    bool,
]


Detector = Callable[
    [BenchmarkSample],
    bool,
]


class BenchmarkRunner:
    """Run clean, poisoned, and defended benchmark scenarios."""

    def __init__(
        self,
        evaluator: Evaluator,
        detector: Detector,
    ) -> None:
        if not callable(evaluator):
            raise BenchmarkRunnerError(
                "evaluator must be callable."
            )

        if not callable(detector):
            raise BenchmarkRunnerError(
                "detector must be callable."
            )

        self._evaluator = evaluator
        self._detector = detector

    def run(
        self,
        dataset: BenchmarkDataset,
    ) -> BenchmarkRunResult:
        """Execute the benchmark."""
        if not isinstance(dataset, BenchmarkDataset):
            raise BenchmarkRunnerError(
                "dataset must be a BenchmarkDataset."
            )

        if dataset.size == 0:
            raise BenchmarkRunnerError(
                "Cannot run benchmark on an empty dataset."
            )

        evaluations: list[SampleEvaluation] = []

        for sample in dataset.samples:
            clean_success = self._evaluator(
                sample,
                sample.clean_code,
                False,
            )

            poisoned_success = self._evaluator(
                sample,
                sample.poisoned_code,
                True,
            )

            defended_success = self._evaluator(
                sample,
                sample.poisoned_code,
                True,
            )

            detected = self._detector(sample)

            evaluations.append(
                SampleEvaluation(
                    sample_id=sample.sample_id,
                    clean_success=bool(clean_success),
                    poisoned_success=bool(poisoned_success),
                    defended_success=bool(defended_success),
                    detected=bool(detected),
                )
            )

        metrics = calculate_metrics(
            clean_success=[
                item.clean_success
                for item in evaluations
            ],
            poisoned_success=[
                item.poisoned_success
                for item in evaluations
            ],
            defended_success=[
                item.defended_success
                for item in evaluations
            ],
            detected=[
                item.detected
                for item in evaluations
            ],
        )

        return BenchmarkRunResult(
            evaluations=tuple(evaluations),
            metrics=metrics,
        )