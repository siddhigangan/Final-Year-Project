"""Persistent storage for SecureCodeRAG experiment results.

This module provides a small, deterministic JSONL-based persistence layer
for experiment and benchmark results.

The result store intentionally does not execute experiments, calculate
metrics, or modify experiment results. Its responsibility is serialization,
persistence, loading, filtering, and basic result discovery.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.benchmark import BenchmarkRun
from src.models import ExperimentCondition, ExperimentResult


class ResultStoreError(RuntimeError):
    """Base exception for result-store failures."""


class ResultStoreConfigurationError(ValueError, ResultStoreError):
    """Raised when result-store configuration is invalid."""


class ResultStoreInputError(ResultStoreError):
    """Raised when result-store input is invalid."""


class ResultStore:
    """Persist and retrieve experiment and benchmark results.

    Results are stored as JSON Lines files:

    ``experiments.jsonl``
        One serialized :class:`ExperimentResult` per line.

    ``benchmark_runs.jsonl``
        One serialized :class:`BenchmarkRun` per line.
    """

    EXPERIMENTS_FILE = "experiments.jsonl"
    BENCHMARK_RUNS_FILE = "benchmark_runs.jsonl"

    def __init__(
        self,
        root_path: str | Path,
        *,
        create: bool = True,
    ) -> None:
        if not isinstance(root_path, (str, Path)):
            raise ResultStoreConfigurationError(
                "root_path must be a string or pathlib.Path."
            )

        root = Path(root_path).expanduser()

        if not root.is_absolute():
            root = root.resolve()

        if root.exists() and not root.is_dir():
            raise ResultStoreConfigurationError(
                f"Result-store path is not a directory: {root}"
            )

        if create:
            root.mkdir(parents=True, exist_ok=True)

        self._root_path = root
        self._experiments_path = root / self.EXPERIMENTS_FILE
        self._benchmark_runs_path = root / self.BENCHMARK_RUNS_FILE

    @property
    def root_path(self) -> Path:
        """Return the result-store root directory."""
        return self._root_path

    @property
    def experiments_path(self) -> Path:
        """Return the experiment-results JSONL path."""
        return self._experiments_path

    @property
    def benchmark_runs_path(self) -> Path:
        """Return the benchmark-runs JSONL path."""
        return self._benchmark_runs_path

    def save_experiment(
        self,
        result: ExperimentResult,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist one experiment result."""
        if not isinstance(result, ExperimentResult):
            raise ResultStoreInputError(
                "result must be an ExperimentResult."
            )

        if not isinstance(overwrite, bool):
            raise ResultStoreInputError(
                "overwrite must be a boolean."
            )

        existing = self._read_jsonl(self._experiments_path)

        serialized = self._experiment_to_dict(result)

        experiment_id = result.experiment_id

        matching_indexes = [
            index
            for index, item in enumerate(existing)
            if item.get("experiment_id") == experiment_id
        ]

        if matching_indexes:
            if not overwrite:
                raise ResultStoreInputError(
                    f"Experiment result already exists: {experiment_id!r}."
                )

            existing[matching_indexes[0]] = serialized

            if len(matching_indexes) > 1:
                duplicate_indexes = set(matching_indexes[1:])
                existing = [
                    item
                    for index, item in enumerate(existing)
                    if index not in duplicate_indexes
                ]

            self._write_jsonl(self._experiments_path, existing)
            return

        self._append_jsonl(self._experiments_path, serialized)

    def save_benchmark_run(
        self,
        run: BenchmarkRun,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist one benchmark run."""
        if not isinstance(run, BenchmarkRun):
            raise ResultStoreInputError(
                "run must be a BenchmarkRun."
            )

        if not isinstance(overwrite, bool):
            raise ResultStoreInputError(
                "overwrite must be a boolean."
            )

        existing = self._read_jsonl(self._benchmark_runs_path)

        run_key = self._benchmark_run_key(run)

        serialized = self._benchmark_run_to_dict(run)

        matching_indexes = [
            index
            for index, item in enumerate(existing)
            if self._stored_benchmark_run_key(item) == run_key
        ]

        if matching_indexes:
            if not overwrite:
                raise ResultStoreInputError(
                    "Benchmark run already exists for "
                    f"{run.benchmark_id!r} with the same run key."
                )

            existing[matching_indexes[0]] = serialized

            if len(matching_indexes) > 1:
                duplicate_indexes = set(matching_indexes[1:])
                existing = [
                    item
                    for index, item in enumerate(existing)
                    if index not in duplicate_indexes
                ]

            self._write_jsonl(self._benchmark_runs_path, existing)
            return

        self._append_jsonl(
            self._benchmark_runs_path,
            serialized,
        )

    def save(
        self,
        result: ExperimentResult | BenchmarkRun,
        *,
        overwrite: bool = False,
    ) -> None:
        """Persist either an experiment result or benchmark run."""
        if isinstance(result, ExperimentResult):
            self.save_experiment(
                result,
                overwrite=overwrite,
            )
            return

        if isinstance(result, BenchmarkRun):
            self.save_benchmark_run(
                result,
                overwrite=overwrite,
            )
            return

        raise ResultStoreInputError(
            "result must be an ExperimentResult or BenchmarkRun."
        )

    def load_experiment(
        self,
        experiment_id: str,
    ) -> ExperimentResult:
        """Load one experiment result by experiment ID."""
        experiment_id = self._validate_identifier(
            experiment_id,
            "experiment_id",
        )

        records = self._read_jsonl(self._experiments_path)

        matches = [
            record
            for record in records
            if record.get("experiment_id") == experiment_id
        ]

        if not matches:
            raise ResultStoreInputError(
                f"Experiment result not found: {experiment_id!r}."
            )

        return self._experiment_from_dict(matches[-1])

    def load_experiments(self) -> tuple[ExperimentResult, ...]:
        """Load all stored experiment results."""
        records = self._read_jsonl(self._experiments_path)

        results = [
            self._experiment_from_dict(record)
            for record in records
        ]

        return tuple(
            sorted(
                results,
                key=lambda result: (
                    result.created_at,
                    result.experiment_id,
                ),
            )
        )

    def load_benchmark_runs(self) -> tuple[BenchmarkRun, ...]:
        """Load all stored benchmark runs."""
        records = self._read_jsonl(self._benchmark_runs_path)

        runs = [
            self._benchmark_run_from_dict(record)
            for record in records
        ]

        return tuple(
            sorted(
                runs,
                key=lambda run: (
                    run.benchmark_id,
                    run.duration_ms,
                ),
            )
        )

    def query_experiments(
        self,
        *,
        experiment_id: str | None = None,
        condition: ExperimentCondition | str | None = None,
        repository_id: str | None = None,
        task_type: str | None = None,
        model_name: str | None = None,
        retriever_name: str | None = None,
        defense_enabled: bool | None = None,
    ) -> tuple[ExperimentResult, ...]:
        """Query stored experiment results using exact filters."""
        if experiment_id is not None:
            experiment_id = self._validate_identifier(
                experiment_id,
                "experiment_id",
            )

        if repository_id is not None:
            repository_id = self._validate_identifier(
                repository_id,
                "repository_id",
            )

        if task_type is not None:
            task_type = self._validate_identifier(
                task_type,
                "task_type",
            )

        if model_name is not None:
            model_name = self._validate_identifier(
                model_name,
                "model_name",
            )

        if retriever_name is not None:
            retriever_name = self._validate_identifier(
                retriever_name,
                "retriever_name",
            )

        if defense_enabled is not None and not isinstance(
            defense_enabled,
            bool,
        ):
            raise ResultStoreInputError(
                "defense_enabled must be a boolean or None."
            )

        normalized_condition = self._normalize_condition(condition)

        results = self.load_experiments()

        filtered = []

        for result in results:
            if (
                experiment_id is not None
                and result.experiment_id != experiment_id
            ):
                continue

            if (
                normalized_condition is not None
                and result.condition != normalized_condition
            ):
                continue

            metadata = result.metadata

            if (
                repository_id is not None
                and metadata.get("repository_id") != repository_id
            ):
                continue

            if (
                task_type is not None
                and metadata.get("task") != task_type
                and metadata.get("task_type") != task_type
            ):
                continue

            if (
                model_name is not None
                and metadata.get("model_name") != model_name
            ):
                continue

            if (
                retriever_name is not None
                and metadata.get("retriever_name") != retriever_name
            ):
                continue

            if defense_enabled is not None:
                stored_defense = metadata.get("defense_enabled")

                if stored_defense is None:
                    stored_defense = (
                        result.condition
                        in {
                            ExperimentCondition.POISONED_DEFENSE,
                            ExperimentCondition.CLEAN_DEFENSE,
                        }
                    )

                if bool(stored_defense) != defense_enabled:
                    continue

            filtered.append(result)

        return tuple(filtered)

    def list_experiment_ids(self) -> tuple[str, ...]:
        """Return all stored experiment IDs in deterministic order."""
        return tuple(
            result.experiment_id
            for result in self.load_experiments()
        )

    def list_benchmark_ids(self) -> tuple[str, ...]:
        """Return all stored benchmark IDs in deterministic order."""
        runs = self.load_benchmark_runs()

        return tuple(
            sorted(
                {run.benchmark_id for run in runs},
            )
        )

    def count_experiments(self) -> int:
        """Return the number of stored experiment results."""
        return len(self._read_jsonl(self._experiments_path))

    def count_benchmark_runs(self) -> int:
        """Return the number of stored benchmark runs."""
        return len(self._read_jsonl(self._benchmark_runs_path))

    def clear_experiments(self) -> None:
        """Remove all persisted experiment results."""
        self._remove_file(self._experiments_path)

    def clear_benchmark_runs(self) -> None:
        """Remove all persisted benchmark runs."""
        self._remove_file(self._benchmark_runs_path)

    def clear(self) -> None:
        """Remove all persisted result data."""
        self.clear_experiments()
        self.clear_benchmark_runs()

    def _append_jsonl(
        self,
        path: Path,
        record: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                handle.write("\n")
        except OSError as exc:
            raise ResultStoreError(
                f"Unable to append result data to {path}."
            ) from exc

    def _write_jsonl(
        self,
        path: Path,
        records: list[dict[str, Any]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    handle.write("\n")
        except OSError as exc:
            raise ResultStoreError(
                f"Unable to write result data to {path}."
            ) from exc

    def _read_jsonl(
        self,
        path: Path,
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return []

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                records = []

                for line_number, line in enumerate(
                    handle,
                    start=1,
                ):
                    stripped = line.strip()

                    if not stripped:
                        continue

                    try:
                        value = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ResultStoreError(
                            "Invalid JSON in result store "
                            f"{path} at line {line_number}."
                        ) from exc

                    if not isinstance(value, dict):
                        raise ResultStoreError(
                            "Each JSONL record must be an object: "
                            f"{path}:{line_number}."
                        )

                    records.append(value)

                return records
        except OSError as exc:
            raise ResultStoreError(
                f"Unable to read result data from {path}."
            ) from exc

    def _remove_file(self, path: Path) -> None:
        if not path.exists():
            return

        try:
            path.unlink()
        except OSError as exc:
            raise ResultStoreError(
                f"Unable to remove result file {path}."
            ) from exc

    @staticmethod
    def _validate_identifier(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise ResultStoreInputError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ResultStoreInputError(
                f"{field_name} cannot be empty."
            )

        return value

    @staticmethod
    def _normalize_condition(
        condition: ExperimentCondition | str | None,
    ) -> ExperimentCondition | None:
        if condition is None:
            return None

        if isinstance(condition, ExperimentCondition):
            return condition

        if isinstance(condition, str):
            try:
                return ExperimentCondition(condition)
            except ValueError as exc:
                raise ResultStoreInputError(
                    f"Unknown experiment condition: {condition!r}."
                ) from exc

        raise ResultStoreInputError(
            "condition must be an ExperimentCondition, string, or None."
        )

    @staticmethod
    def _experiment_to_dict(
        result: ExperimentResult,
    ) -> dict[str, Any]:
        return {
            "record_type": "experiment_result",
            "experiment_id": result.experiment_id,
            "condition": result.condition.value,
            "request_id": result.request_id,
            "poison_present": result.poison_present,
            "poison_retrieved": result.poison_retrieved,
            "poison_in_context": result.poison_in_context,
            "generation_changed": result.generation_changed,
            "vulnerability_introduced": result.vulnerability_introduced,
            "poison_retrieval_rate": result.poison_retrieval_rate,
            "attack_success_rate": result.attack_success_rate,
            "vulnerability_introduction_rate": (
                result.vulnerability_introduction_rate
            ),
            "utility_score": result.utility_score,
            "latency_ms": result.latency_ms,
            "findings": [
                finding.as_dict()
                if hasattr(finding, "as_dict")
                else {
                    "finding_id": finding.finding_id,
                    "rule_id": finding.rule_id,
                    "title": finding.title,
                    "description": finding.description,
                    "severity": finding.severity.value,
                    "decision": finding.decision.value,
                    "file_path": finding.file_path,
                    "start_line": finding.start_line,
                    "end_line": finding.end_line,
                    "evidence": finding.evidence,
                    "analyzer": finding.analyzer,
                    "confidence": finding.confidence,
                    "metadata": dict(finding.metadata),
                }
                for finding in result.findings
            ],
            "metadata": dict(result.metadata),
            "created_at": result.created_at.isoformat(),
        }

    @staticmethod
    def _experiment_from_dict(
        data: dict[str, Any],
    ) -> ExperimentResult:
        if data.get("record_type") not in {
            None,
            "experiment_result",
        }:
            raise ResultStoreError(
                "Invalid experiment result record type."
            )

        try:
            condition = ExperimentCondition(data["condition"])

            created_at = datetime.fromisoformat(
                data["created_at"]
            )

            return ExperimentResult(
                experiment_id=data["experiment_id"],
                condition=condition,
                request_id=data["request_id"],
                poison_present=bool(data["poison_present"]),
                poison_retrieved=bool(data["poison_retrieved"]),
                poison_in_context=bool(data["poison_in_context"]),
                generation_changed=bool(data["generation_changed"]),
                vulnerability_introduced=bool(
                    data["vulnerability_introduced"]
                ),
                poison_retrieval_rate=float(
                    data["poison_retrieval_rate"]
                ),
                attack_success_rate=float(
                    data["attack_success_rate"]
                ),
                vulnerability_introduction_rate=float(
                    data["vulnerability_introduction_rate"]
                ),
                utility_score=float(data["utility_score"]),
                latency_ms=float(data["latency_ms"]),
                findings=[],
                metadata=dict(data.get("metadata", {})),
                created_at=created_at,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultStoreError(
                "Invalid serialized ExperimentResult."
            ) from exc

    @staticmethod
    def _benchmark_run_key(
        run: BenchmarkRun,
    ) -> str:
        metadata = run.metadata

        created_at = metadata.get("created_at")

        if created_at is None:
            created_at = metadata.get("run_id")

        if created_at is None:
            created_at = (
                f"{run.total_cases}:"
                f"{run.completed_cases}:"
                f"{run.failed_cases}:"
                f"{run.duration_ms:.6f}"
            )

        return (
            f"{run.benchmark_id}|"
            f"{run.benchmark_name}|"
            f"{created_at}"
        )

    @staticmethod
    def _stored_benchmark_run_key(
        data: dict[str, Any],
    ) -> str:
        benchmark_id = str(data.get("benchmark_id", ""))
        benchmark_name = str(data.get("benchmark_name", ""))

        metadata = data.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        created_at = metadata.get("created_at")

        if created_at is None:
            created_at = metadata.get("run_id")

        if created_at is None:
            created_at = (
                f"{data.get('total_cases', 0)}:"
                f"{data.get('completed_cases', 0)}:"
                f"{data.get('failed_cases', 0)}:"
                f"{float(data.get('duration_ms', 0.0)):.6f}"
            )

        return (
            f"{benchmark_id}|"
            f"{benchmark_name}|"
            f"{created_at}"
        )

    @staticmethod
    def _benchmark_run_to_dict(
        run: BenchmarkRun,
    ) -> dict[str, Any]:
        return {
            "record_type": "benchmark_run",
            "benchmark_id": run.benchmark_id,
            "benchmark_name": run.benchmark_name,
            "total_cases": run.total_cases,
            "completed_cases": run.completed_cases,
            "failed_cases": run.failed_cases,
            "duration_ms": run.duration_ms,
            "results": [
                ResultStore._experiment_to_dict(result)
                for result in run.results
            ],
            "failures": [
                dict(failure)
                for failure in run.failures
            ],
            "metadata": {
                **dict(run.metadata),
                "stored_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            },
        }

    @staticmethod
    def _benchmark_run_from_dict(
        data: dict[str, Any],
    ) -> BenchmarkRun:
        if data.get("record_type") not in {
            None,
            "benchmark_run",
        }:
            raise ResultStoreError(
                "Invalid benchmark run record type."
            )

        try:
            results = tuple(
                ResultStore._experiment_from_dict(
                    record
                )
                for record in data.get("results", [])
            )

            failures = tuple(
                dict(failure)
                for failure in data.get("failures", [])
            )

            return BenchmarkRun(
                benchmark_id=data["benchmark_id"],
                benchmark_name=data["benchmark_name"],
                total_cases=int(data["total_cases"]),
                completed_cases=int(data["completed_cases"]),
                failed_cases=int(data["failed_cases"]),
                results=results,
                failures=failures,
                duration_ms=float(data["duration_ms"]),
                metadata=dict(data.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultStoreError(
                "Invalid serialized BenchmarkRun."
            ) from exc