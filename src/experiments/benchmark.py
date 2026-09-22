"""Benchmark definitions and execution for controlled CodeRAG experiments.

This module provides a thin benchmark orchestration layer above
:mod:`src.experiments.experiment_runner`.

The benchmark is responsible for defining, validating, ordering, and
executing experiment cases. Retrieval, defense, generation, security
analysis, and experiment-level metric computation remain responsibilities
of the existing experiment pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from src.experiments.experiment_runner import (
    ExperimentRunInput,
    ExperimentRunner,
    ExperimentRunnerError,
)
from src.models import ExperimentCondition, ExperimentResult


class BenchmarkError(RuntimeError):
    """Base exception for benchmark-related failures."""


class BenchmarkConfigurationError(ValueError, BenchmarkError):
    """Raised when benchmark configuration is invalid."""


class BenchmarkInputError(BenchmarkError):
    """Raised when benchmark input data is invalid."""


@dataclass(frozen=True)
class BenchmarkCase:
    """One reproducible experiment case belonging to a benchmark."""

    case_id: str
    experiment: ExperimentRunInput
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise BenchmarkInputError(
                "case_id must be a non-empty string."
            )

        if not isinstance(self.experiment, ExperimentRunInput):
            raise BenchmarkInputError(
                "experiment must be an ExperimentRunInput."
            )

        if not isinstance(self.metadata, dict):
            raise BenchmarkInputError(
                "metadata must be a dictionary."
            )

    @property
    def experiment_id(self) -> str:
        """Return the underlying experiment identifier."""

        return self.experiment.experiment_config.experiment_id

    @property
    def condition(self) -> ExperimentCondition:
        """Return the experiment condition."""

        return self.experiment.experiment_config.condition

    @property
    def seed(self) -> int:
        """Return the reproducibility seed."""

        return self.experiment.experiment_config.seed


@dataclass(frozen=True)
class BenchmarkRun:
    """Structured result of executing a benchmark."""

    benchmark_id: str
    benchmark_name: str
    total_cases: int
    completed_cases: int
    failed_cases: int
    results: tuple[ExperimentResult, ...]
    failures: tuple[dict[str, Any], ...] = ()
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_id, str):
            raise BenchmarkInputError(
                "benchmark_id must be a string."
            )

        if not self.benchmark_id.strip():
            raise BenchmarkInputError(
                "benchmark_id cannot be empty."
            )

        if not isinstance(self.benchmark_name, str):
            raise BenchmarkInputError(
                "benchmark_name must be a string."
            )

        if self.total_cases < 0:
            raise BenchmarkInputError(
                "total_cases cannot be negative."
            )

        if self.completed_cases < 0:
            raise BenchmarkInputError(
                "completed_cases cannot be negative."
            )

        if self.failed_cases < 0:
            raise BenchmarkInputError(
                "failed_cases cannot be negative."
            )

        if (
            self.completed_cases + self.failed_cases
            > self.total_cases
        ):
            raise BenchmarkInputError(
                "completed_cases + failed_cases cannot exceed "
                "total_cases."
            )

        if self.duration_ms < 0:
            raise BenchmarkInputError(
                "duration_ms cannot be negative."
            )

        if not isinstance(self.results, tuple):
            raise BenchmarkInputError(
                "results must be a tuple."
            )

        if not isinstance(self.failures, tuple):
            raise BenchmarkInputError(
                "failures must be a tuple."
            )

        if not isinstance(self.metadata, dict):
            raise BenchmarkInputError(
                "metadata must be a dictionary."
            )

        for result in self.results:
            if not isinstance(result, ExperimentResult):
                raise BenchmarkInputError(
                    "results must contain only ExperimentResult objects."
                )

        for failure in self.failures:
            if not isinstance(failure, dict):
                raise BenchmarkInputError(
                    "failures must contain only dictionaries."
                )

    @property
    def success_rate(self) -> float:
        """Return the fraction of cases that completed successfully."""

        if self.total_cases == 0:
            return 0.0

        return self.completed_cases / self.total_cases

    @property
    def has_failures(self) -> bool:
        """Return whether at least one benchmark case failed."""

        return self.failed_cases > 0

    @property
    def is_complete(self) -> bool:
        """Return whether every benchmark case completed or failed."""

        return (
            self.completed_cases + self.failed_cases
            == self.total_cases
        )


class Benchmark:
    """Manage and execute a deterministic collection of benchmark cases."""

    DEFAULT_NAME = "CodeRAG-PoisonBench"
    DEFAULT_VERSION = "0.1.0"

    def __init__(
        self,
        *,
        benchmark_id: str,
        name: str = DEFAULT_NAME,
        version: str = DEFAULT_VERSION,
        description: str = "",
        cases: Iterable[BenchmarkCase] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise BenchmarkConfigurationError(
                "benchmark_id must be a non-empty string."
            )

        if not isinstance(name, str) or not name.strip():
            raise BenchmarkConfigurationError(
                "name must be a non-empty string."
            )

        if not isinstance(version, str) or not version.strip():
            raise BenchmarkConfigurationError(
                "version must be a non-empty string."
            )

        if not isinstance(description, str):
            raise BenchmarkConfigurationError(
                "description must be a string."
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise BenchmarkConfigurationError(
                "metadata must be a dictionary."
            )

        self._benchmark_id = benchmark_id.strip()
        self._name = name.strip()
        self._version = version.strip()
        self._description = description.strip()
        self._metadata = dict(metadata or {})
        self._cases: dict[str, BenchmarkCase] = {}

        if cases is not None:
            self.add_cases(cases)

    @property
    def benchmark_id(self) -> str:
        """Return the benchmark identifier."""

        return self._benchmark_id

    @property
    def name(self) -> str:
        """Return the benchmark name."""

        return self._name

    @property
    def version(self) -> str:
        """Return the benchmark version."""

        return self._version

    @property
    def description(self) -> str:
        """Return the benchmark description."""

        return self._description

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of benchmark metadata."""

        return dict(self._metadata)

    @property
    def case_count(self) -> int:
        """Return the number of registered cases."""

        return len(self._cases)

    @property
    def cases(self) -> tuple[BenchmarkCase, ...]:
        """Return benchmark cases in deterministic order."""

        return tuple(
            self._cases[case_id]
            for case_id in sorted(self._cases)
        )

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        """Return registered experiment identifiers."""

        return tuple(
            case.experiment_id
            for case in self.cases
        )

    def add_case(self, case: BenchmarkCase) -> None:
        """Register one benchmark case."""

        if not isinstance(case, BenchmarkCase):
            raise BenchmarkInputError(
                "case must be a BenchmarkCase."
            )

        if case.case_id in self._cases:
            raise BenchmarkInputError(
                f"Duplicate benchmark case ID: {case.case_id!r}."
            )

        if case.experiment_id in self.experiment_ids:
            raise BenchmarkInputError(
                "Duplicate experiment ID: "
                f"{case.experiment_id!r}."
            )

        self._validate_case_compatibility(case)
        self._cases[case.case_id] = case

    def add_cases(
        self,
        cases: Iterable[BenchmarkCase],
    ) -> None:
        """Register multiple benchmark cases."""

        if isinstance(cases, (str, bytes)):
            raise BenchmarkInputError(
                "cases must be an iterable of BenchmarkCase objects."
            )

        for case in cases:
            self.add_case(case)

    def get_case(self, case_id: str) -> BenchmarkCase:
        """Return a registered case by ID."""

        if not isinstance(case_id, str) or not case_id.strip():
            raise BenchmarkInputError(
                "case_id must be a non-empty string."
            )

        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise BenchmarkInputError(
                f"Unknown benchmark case: {case_id!r}."
            ) from exc

    def remove_case(self, case_id: str) -> BenchmarkCase:
        """Remove and return a registered case."""

        if not isinstance(case_id, str) or not case_id.strip():
            raise BenchmarkInputError(
                "case_id must be a non-empty string."
            )

        try:
            return self._cases.pop(case_id)
        except KeyError as exc:
            raise BenchmarkInputError(
                f"Unknown benchmark case: {case_id!r}."
            ) from exc

    def clear(self) -> None:
        """Remove all registered benchmark cases."""

        self._cases.clear()

    def validate(self) -> None:
        """Validate the complete benchmark definition."""

        if not self._benchmark_id:
            raise BenchmarkConfigurationError(
                "benchmark_id cannot be empty."
            )

        if not self._name:
            raise BenchmarkConfigurationError(
                "name cannot be empty."
            )

        if not self._version:
            raise BenchmarkConfigurationError(
                "version cannot be empty."
            )

        experiment_ids: set[str] = set()

        for case in self.cases:
            if case.experiment_id in experiment_ids:
                raise BenchmarkInputError(
                    "Duplicate experiment ID: "
                    f"{case.experiment_id!r}."
                )

            experiment_ids.add(case.experiment_id)
            self._validate_case_compatibility(case)

    def run(
        self,
        runner: ExperimentRunner,
        *,
        stop_on_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkRun:
        """Execute every registered benchmark case.

        Cases are executed in deterministic case-ID order. A failed case is
        recorded in the returned ``BenchmarkRun``. When ``stop_on_error`` is
        enabled, execution stops after the first failure.
        """

        if not isinstance(runner, ExperimentRunner):
            raise BenchmarkInputError(
                "runner must be an ExperimentRunner."
            )

        if not isinstance(stop_on_error, bool):
            raise BenchmarkInputError(
                "stop_on_error must be a boolean."
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise BenchmarkInputError(
                "metadata must be a dictionary."
            )

        self.validate()

        started = perf_counter()
        results: list[ExperimentResult] = []
        failures: list[dict[str, Any]] = []

        for case in self.cases:
            try:
                result = runner.run(case.experiment)

                if not isinstance(result, ExperimentResult):
                    raise BenchmarkError(
                        "ExperimentRunner returned an invalid result "
                        f"for case {case.case_id!r}."
                    )

                results.append(result)

            except (
                ExperimentRunnerError,
                RuntimeError,
                ValueError,
                TypeError,
                OSError,
            ) as exc:
                failure = {
                    "case_id": case.case_id,
                    "experiment_id": case.experiment_id,
                    "condition": case.condition.value,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)

                if stop_on_error:
                    break

        duration_ms = (perf_counter() - started) * 1000.0

        completed_cases = len(results)
        failed_cases = len(failures)

        run_metadata = {
            **self._metadata,
            **(metadata or {}),
            "benchmark_id": self._benchmark_id,
            "benchmark_name": self._name,
            "benchmark_version": self._version,
            "case_order": [
                case.case_id
                for case in self.cases
            ],
            "stop_on_error": stop_on_error,
        }

        return BenchmarkRun(
            benchmark_id=self._benchmark_id,
            benchmark_name=self._name,
            total_cases=self.case_count,
            completed_cases=completed_cases,
            failed_cases=failed_cases,
            results=tuple(results),
            failures=tuple(failures),
            duration_ms=duration_ms,
            metadata=run_metadata,
        )

    def summary(self) -> dict[str, Any]:
        """Return a serializable benchmark definition summary."""

        self.validate()

        conditions: dict[str, int] = {}

        for case in self.cases:
            condition = case.condition.value
            conditions[condition] = conditions.get(condition, 0) + 1

        return {
            "benchmark_id": self._benchmark_id,
            "name": self._name,
            "version": self._version,
            "description": self._description,
            "case_count": self.case_count,
            "conditions": conditions,
            "experiment_ids": list(self.experiment_ids),
            "metadata": dict(self._metadata),
        }

    def _validate_case_compatibility(
        self,
        case: BenchmarkCase,
    ) -> None:
        """Validate fields required for benchmark execution."""

        config = case.experiment.experiment_config

        if not isinstance(config.condition, ExperimentCondition):
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an invalid condition."
            )

        if config.top_k < 1:
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has invalid top_k."
            )

        if config.seed < 0:
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an invalid seed."
            )

        if not config.repository_id.strip():
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an empty repository_id."
            )

        if not config.task_type.strip():
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an empty task_type."
            )

        if not config.model_name.strip():
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an empty model_name."
            )

        if not config.retriever_name.strip():
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an empty retriever_name."
            )

        if not case.experiment.query.strip():
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an empty query."
            )

        if not case.experiment.task.strip():
            raise BenchmarkInputError(
                f"Case {case.case_id!r} has an empty task."
            )

    def __len__(self) -> int:
        """Return the number of benchmark cases."""

        return self.case_count

    def __iter__(self):
        """Iterate over cases in deterministic order."""

        return iter(self.cases)

    def __repr__(self) -> str:
        """Return a concise benchmark representation."""

        return (
            "Benchmark("
            f"benchmark_id={self._benchmark_id!r}, "
            f"name={self._name!r}, "
            f"version={self._version!r}, "
            f"cases={self.case_count}"
            ")"
        )