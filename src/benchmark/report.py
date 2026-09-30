"""Benchmark reporting utilities for SecureCodeRAG."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.benchmark.runner import BenchmarkRunResult


class BenchmarkReportError(RuntimeError):
    """Raised when benchmark report generation fails."""


def _validate_run_result(
    result: BenchmarkRunResult,
) -> None:
    """Validate a benchmark result before creating a report."""
    if not isinstance(result, BenchmarkRunResult):
        raise BenchmarkReportError(
            "result must be a BenchmarkRunResult."
        )

    if result.sample_count <= 0:
        raise BenchmarkReportError(
            "Cannot generate a report from an empty benchmark result."
        )


def build_report(
    result: BenchmarkRunResult,
    *,
    experiment_id: str | None = None,
    condition: str | None = None,
    dataset_version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable benchmark report.

    Args:
        result:
            Completed BenchmarkRunResult.

        experiment_id:
            Optional experiment identifier.

        condition:
            Optional benchmark condition such as clean or poisoned.

        dataset_version:
            Optional dataset version.

        metadata:
            Optional additional metadata.

    Returns:
        A dictionary containing report metadata, metrics,
        and per-sample evaluation results.

    Raises:
        BenchmarkReportError:
            If the supplied result or metadata is invalid.
    """
    _validate_run_result(result)

    if (
        experiment_id is not None
        and (
            not isinstance(experiment_id, str)
            or not experiment_id.strip()
        )
    ):
        raise BenchmarkReportError(
            "experiment_id must be a non-empty string."
        )

    if (
        condition is not None
        and (
            not isinstance(condition, str)
            or not condition.strip()
        )
    ):
        raise BenchmarkReportError(
            "condition must be a non-empty string."
        )

    if (
        dataset_version is not None
        and (
            not isinstance(dataset_version, str)
            or not dataset_version.strip()
        )
    ):
        raise BenchmarkReportError(
            "dataset_version must be a non-empty string."
        )

    if metadata is not None and not isinstance(metadata, dict):
        raise BenchmarkReportError(
            "metadata must be a dictionary."
        )

    report: dict[str, Any] = {
        "report_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sample_count": result.sample_count,
        "metrics": result.metrics.to_dict(),
        "evaluations": [
            evaluation.to_dict()
            for evaluation in result.evaluations
        ],
    }

    if experiment_id is not None:
        report["experiment_id"] = experiment_id

    if condition is not None:
        report["condition"] = condition

    if dataset_version is not None:
        report["dataset_version"] = dataset_version

    if metadata is not None:
        report["metadata"] = dict(metadata)

    return report


def save_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save a benchmark report as formatted JSON.

    Parent directories are created automatically.

    Args:
        report:
            Report dictionary returned by build_report().

        output_path:
            Destination JSON file.

    Returns:
        Absolute Path of the saved report.

    Raises:
        BenchmarkReportError:
            If the report or destination is invalid or
            the file cannot be written.
    """
    if not isinstance(report, dict):
        raise BenchmarkReportError(
            "report must be a dictionary."
        )

    if not report:
        raise BenchmarkReportError(
            "report cannot be empty."
        )

    path = Path(output_path)

    if not str(path).strip():
        raise BenchmarkReportError(
            "output_path must not be empty."
        )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

    except OSError as exc:
        raise BenchmarkReportError(
            f"Failed to save benchmark report: {exc}"
        ) from exc

    return path.resolve()


__all__ = [
    "BenchmarkReportError",
    "build_report",
    "save_report",
]