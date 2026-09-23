"""Research reporting and aggregation for SecureCodeRAG experiments.

This module consumes already-executed ExperimentResult objects and produces
deterministic aggregate reports.

It does not execute experiments, modify stored results, or fabricate missing
metrics. All reported values are derived directly from the supplied results.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from src.models import ExperimentCondition, ExperimentResult


class ReportingError(RuntimeError):
    """Base exception for reporting failures."""


class ReportingInputError(ReportingError):
    """Raised when reporting input is invalid."""


class ReportingConfigurationError(ValueError, ReportingError):
    """Raised when reporting configuration is invalid."""


@dataclass(frozen=True)
class ConditionMetrics:
    """Aggregate metrics for one experiment condition."""

    condition: ExperimentCondition
    sample_count: int

    poison_retrieval_rate: float
    attack_success_rate: float
    vulnerability_introduction_rate: float
    utility_score: float
    latency_ms: float

    generation_change_rate: float
    vulnerability_rate: float

    security_finding_count: int
    blocking_finding_count: int

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def defense_enabled(self) -> bool:
        """Return whether this condition uses the defense pipeline."""
        return self.condition in {
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        }

    @property
    def poisoned(self) -> bool:
        """Return whether this condition contains poisoning."""
        return self.condition in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "condition": self.condition.value,
            "sample_count": self.sample_count,
            "poison_retrieval_rate": self.poison_retrieval_rate,
            "attack_success_rate": self.attack_success_rate,
            "vulnerability_introduction_rate": (
                self.vulnerability_introduction_rate
            ),
            "utility_score": self.utility_score,
            "latency_ms": self.latency_ms,
            "generation_change_rate": self.generation_change_rate,
            "vulnerability_rate": self.vulnerability_rate,
            "security_finding_count": self.security_finding_count,
            "blocking_finding_count": self.blocking_finding_count,
            "defense_enabled": self.defense_enabled,
            "poisoned": self.poisoned,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConditionComparison:
    """Difference between two condition aggregates."""

    baseline: ExperimentCondition
    comparison: ExperimentCondition

    poison_retrieval_rate_delta: float
    attack_success_rate_delta: float
    vulnerability_introduction_rate_delta: float
    utility_score_delta: float
    latency_ms_delta: float
    generation_change_rate_delta: float
    vulnerability_rate_delta: float

    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "baseline": self.baseline.value,
            "comparison": self.comparison.value,
            "poison_retrieval_rate_delta": (
                self.poison_retrieval_rate_delta
            ),
            "attack_success_rate_delta": (
                self.attack_success_rate_delta
            ),
            "vulnerability_introduction_rate_delta": (
                self.vulnerability_introduction_rate_delta
            ),
            "utility_score_delta": self.utility_score_delta,
            "latency_ms_delta": self.latency_ms_delta,
            "generation_change_rate_delta": (
                self.generation_change_rate_delta
            ),
            "vulnerability_rate_delta": (
                self.vulnerability_rate_delta
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SecurityUtilityAnalysis:
    """Security–utility analysis derived from condition metrics."""

    clean_baseline: ExperimentCondition
    poisoned_baseline: ExperimentCondition
    defended_poisoned: ExperimentCondition
    defended_clean: ExperimentCondition

    attack_success_reduction: float
    vulnerability_reduction: float
    poison_retrieval_reduction: float

    defended_poisoned_utility_delta: float
    defended_clean_utility_delta: float

    defended_poisoned_latency_delta_ms: float
    defended_clean_latency_delta_ms: float

    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "clean_baseline": self.clean_baseline.value,
            "poisoned_baseline": self.poisoned_baseline.value,
            "defended_poisoned": self.defended_poisoned.value,
            "defended_clean": self.defended_clean.value,
            "attack_success_reduction": (
                self.attack_success_reduction
            ),
            "vulnerability_reduction": self.vulnerability_reduction,
            "poison_retrieval_reduction": (
                self.poison_retrieval_reduction
            ),
            "defended_poisoned_utility_delta": (
                self.defended_poisoned_utility_delta
            ),
            "defended_clean_utility_delta": (
                self.defended_clean_utility_delta
            ),
            "defended_poisoned_latency_delta_ms": (
                self.defended_poisoned_latency_delta_ms
            ),
            "defended_clean_latency_delta_ms": (
                self.defended_clean_latency_delta_ms
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExperimentReport:
    """Complete aggregate report for a collection of experiments."""

    total_experiments: int
    conditions: tuple[ConditionMetrics, ...]
    comparisons: tuple[ConditionComparison, ...]
    security_utility: SecurityUtilityAnalysis | None

    total_security_findings: int
    total_blocking_findings: int

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def condition_count(self) -> int:
        """Return the number of conditions represented."""
        return len(self.conditions)

    @property
    def has_all_conditions(self) -> bool:
        """Return whether A/B/C/D are all represented."""
        present = {
            metrics.condition
            for metrics in self.conditions
        }

        return present == set(ExperimentCondition)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return {
            "total_experiments": self.total_experiments,
            "conditions": [
                metrics.as_dict()
                for metrics in self.conditions
            ],
            "comparisons": [
                comparison.as_dict()
                for comparison in self.comparisons
            ],
            "security_utility": (
                self.security_utility.as_dict()
                if self.security_utility is not None
                else None
            ),
            "total_security_findings": (
                self.total_security_findings
            ),
            "total_blocking_findings": (
                self.total_blocking_findings
            ),
            "metadata": dict(self.metadata),
        }


class ExperimentReporter:
    """Aggregate and compare executed experiment results."""

    def __init__(
        self,
        results: Iterable[ExperimentResult] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if metadata is not None and not isinstance(
            metadata,
            dict,
        ):
            raise ReportingConfigurationError(
                "metadata must be a dictionary."
            )

        self._results: list[ExperimentResult] = []
        self._metadata = dict(metadata or {})

        if results is not None:
            self.add_results(results)

    @property
    def results(self) -> tuple[ExperimentResult, ...]:
        """Return results in insertion order."""
        return tuple(self._results)

    @property
    def result_count(self) -> int:
        """Return the number of supplied results."""
        return len(self._results)

    def add_result(
        self,
        result: ExperimentResult,
    ) -> None:
        """Add one experiment result."""
        if not isinstance(result, ExperimentResult):
            raise ReportingInputError(
                "result must be an ExperimentResult."
            )

        if any(
            existing.experiment_id == result.experiment_id
            for existing in self._results
        ):
            raise ReportingInputError(
                "Duplicate experiment ID: "
                f"{result.experiment_id!r}."
            )

        self._results.append(result)

    def add_results(
        self,
        results: Iterable[ExperimentResult],
    ) -> None:
        """Add multiple experiment results."""
        if isinstance(results, (str, bytes)):
            raise ReportingInputError(
                "results must be an iterable of ExperimentResult."
            )

        for result in results:
            self.add_result(result)

    def clear(self) -> None:
        """Remove all currently loaded results."""
        self._results.clear()

    def condition_metrics(
        self,
        condition: ExperimentCondition,
    ) -> ConditionMetrics:
        """Aggregate results for one condition."""
        if not isinstance(
            condition,
            ExperimentCondition,
        ):
            raise ReportingInputError(
                "condition must be an ExperimentCondition."
            )

        matching = [
            result
            for result in self._results
            if result.condition == condition
        ]

        if not matching:
            raise ReportingInputError(
                "No experiment results available for "
                f"condition {condition.value!r}."
            )

        return self._aggregate_condition(
            condition,
            matching,
        )

    def all_condition_metrics(
        self,
    ) -> tuple[ConditionMetrics, ...]:
        """Aggregate every condition represented in the data."""
        grouped: dict[
            ExperimentCondition,
            list[ExperimentResult],
        ] = {}

        for result in self._results:
            grouped.setdefault(
                result.condition,
                [],
            ).append(result)

        return tuple(
            self._aggregate_condition(
                condition,
                grouped[condition],
            )
            for condition in ExperimentCondition
            if condition in grouped
        )

    def compare_conditions(
        self,
        baseline: ExperimentCondition,
        comparison: ExperimentCondition,
    ) -> ConditionComparison:
        """Compare two available experiment conditions."""
        baseline_metrics = self.condition_metrics(
            baseline,
        )
        comparison_metrics = self.condition_metrics(
            comparison,
        )

        return ConditionComparison(
            baseline=baseline,
            comparison=comparison,
            poison_retrieval_rate_delta=(
                comparison_metrics.poison_retrieval_rate
                - baseline_metrics.poison_retrieval_rate
            ),
            attack_success_rate_delta=(
                comparison_metrics.attack_success_rate
                - baseline_metrics.attack_success_rate
            ),
            vulnerability_introduction_rate_delta=(
                comparison_metrics.vulnerability_introduction_rate
                - baseline_metrics.vulnerability_introduction_rate
            ),
            utility_score_delta=(
                comparison_metrics.utility_score
                - baseline_metrics.utility_score
            ),
            latency_ms_delta=(
                comparison_metrics.latency_ms
                - baseline_metrics.latency_ms
            ),
            generation_change_rate_delta=(
                comparison_metrics.generation_change_rate
                - baseline_metrics.generation_change_rate
            ),
            vulnerability_rate_delta=(
                comparison_metrics.vulnerability_rate
                - baseline_metrics.vulnerability_rate
            ),
        )

    def security_utility_analysis(
        self,
    ) -> SecurityUtilityAnalysis:
        """Calculate the A/B/C/D security–utility analysis.

        This method requires all four experimental conditions.
        """
        metrics = {
            condition_metrics.condition: condition_metrics
            for condition_metrics in self.all_condition_metrics()
        }

        required = set(ExperimentCondition)

        missing = required - set(metrics)

        if missing:
            missing_values = ", ".join(
                sorted(
                    condition.value
                    for condition in missing
                )
            )

            raise ReportingInputError(
                "Security–utility analysis requires all four "
                f"conditions. Missing: {missing_values}."
            )

        clean_baseline = metrics[
            ExperimentCondition.CLEAN_NO_DEFENSE
        ]
        poisoned_baseline = metrics[
            ExperimentCondition.POISONED_NO_DEFENSE
        ]
        defended_poisoned = metrics[
            ExperimentCondition.POISONED_DEFENSE
        ]
        defended_clean = metrics[
            ExperimentCondition.CLEAN_DEFENSE
        ]

        return SecurityUtilityAnalysis(
            clean_baseline=(
                ExperimentCondition.CLEAN_NO_DEFENSE
            ),
            poisoned_baseline=(
                ExperimentCondition.POISONED_NO_DEFENSE
            ),
            defended_poisoned=(
                ExperimentCondition.POISONED_DEFENSE
            ),
            defended_clean=(
                ExperimentCondition.CLEAN_DEFENSE
            ),
            attack_success_reduction=(
                poisoned_baseline.attack_success_rate
                - defended_poisoned.attack_success_rate
            ),
            vulnerability_reduction=(
                poisoned_baseline.vulnerability_introduction_rate
                - defended_poisoned.vulnerability_introduction_rate
            ),
            poison_retrieval_reduction=(
                poisoned_baseline.poison_retrieval_rate
                - defended_poisoned.poison_retrieval_rate
            ),
            defended_poisoned_utility_delta=(
                defended_poisoned.utility_score
                - poisoned_baseline.utility_score
            ),
            defended_clean_utility_delta=(
                defended_clean.utility_score
                - clean_baseline.utility_score
            ),
            defended_poisoned_latency_delta_ms=(
                defended_poisoned.latency_ms
                - poisoned_baseline.latency_ms
            ),
            defended_clean_latency_delta_ms=(
                defended_clean.latency_ms
                - clean_baseline.latency_ms
            ),
        )

    def build_report(self) -> ExperimentReport:
        """Build the complete aggregate experiment report."""
        condition_metrics = self.all_condition_metrics()

        comparisons: list[ConditionComparison] = []

        available = {
            metrics.condition
            for metrics in condition_metrics
        }

        comparison_pairs = (
            (
                ExperimentCondition.CLEAN_NO_DEFENSE,
                ExperimentCondition.POISONED_NO_DEFENSE,
            ),
            (
                ExperimentCondition.POISONED_NO_DEFENSE,
                ExperimentCondition.POISONED_DEFENSE,
            ),
            (
                ExperimentCondition.CLEAN_NO_DEFENSE,
                ExperimentCondition.CLEAN_DEFENSE,
            ),
        )

        for baseline, comparison in comparison_pairs:
            if baseline in available and comparison in available:
                comparisons.append(
                    self.compare_conditions(
                        baseline,
                        comparison,
                    )
                )

        security_utility = None

        if available == set(ExperimentCondition):
            security_utility = (
                self.security_utility_analysis()
            )

        total_findings = sum(
            len(result.findings)
            for result in self._results
        )

        total_blocking = sum(
            self._blocking_finding_count(result)
            for result in self._results
        )

        return ExperimentReport(
            total_experiments=len(self._results),
            conditions=condition_metrics,
            comparisons=tuple(comparisons),
            security_utility=security_utility,
            total_security_findings=total_findings,
            total_blocking_findings=total_blocking,
            metadata={
                **self._metadata,
                "report_type": "experiment_report",
                "condition_count": len(condition_metrics),
            },
        )

    @staticmethod
    def _aggregate_condition(
        condition: ExperimentCondition,
        results: list[ExperimentResult],
    ) -> ConditionMetrics:
        if not results:
            raise ReportingInputError(
                "Cannot aggregate an empty result set."
            )

        sample_count = len(results)

        return ConditionMetrics(
            condition=condition,
            sample_count=sample_count,
            poison_retrieval_rate=ExperimentReporter._mean(
                result.poison_retrieval_rate
                for result in results
            ),
            attack_success_rate=ExperimentReporter._mean(
                result.attack_success_rate
                for result in results
            ),
            vulnerability_introduction_rate=(
                ExperimentReporter._mean(
                    result.vulnerability_introduction_rate
                    for result in results
                )
            ),
            utility_score=ExperimentReporter._mean(
                result.utility_score
                for result in results
            ),
            latency_ms=ExperimentReporter._mean(
                result.latency_ms
                for result in results
            ),
            generation_change_rate=ExperimentReporter._mean(
                1.0 if result.generation_changed else 0.0
                for result in results
            ),
            vulnerability_rate=ExperimentReporter._mean(
                1.0 if result.vulnerability_introduced else 0.0
                for result in results
            ),
            security_finding_count=sum(
                len(result.findings)
                for result in results
            ),
            blocking_finding_count=sum(
                ExperimentReporter._blocking_finding_count(
                    result
                )
                for result in results
            ),
            metadata={
                "experiment_ids": [
                    result.experiment_id
                    for result in results
                ],
            },
        )

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values_list = list(values)

        if not values_list:
            raise ReportingInputError(
                "Cannot calculate the mean of empty values."
            )

        return float(fmean(values_list))

    @staticmethod
    def _blocking_finding_count(
        result: ExperimentResult,
    ) -> int:
        count = 0

        for finding in result.findings:
            decision = getattr(
                finding,
                "decision",
                None,
            )

            decision_value = getattr(
                decision,
                "value",
                decision,
            )

            if decision_value == "reject":
                count += 1

        return count

    def __len__(self) -> int:
        return self.result_count

    def __iter__(self):
        return iter(self._results)

    def __repr__(self) -> str:
        return (
            "ExperimentReporter("
            f"results={self.result_count}"
            ")"
        )