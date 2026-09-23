"""Benchmark package for SecureCodeRAG."""

from src.benchmark.dataset import (
    BenchmarkDataset,
    BenchmarkDatasetError,
    BenchmarkDatasetInputError,
    BenchmarkSample,
)
from src.benchmark.metrics import (
    BenchmarkMetricError,
    MetricResult,
    calculate_metrics,
)
from src.benchmark.report import (
    BenchmarkReportError,
    build_report,
    save_report,
)
from src.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkRunnerError,
    BenchmarkRunResult,
    SampleEvaluation,
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkDatasetError",
    "BenchmarkDatasetInputError",
    "BenchmarkMetricError",
    "BenchmarkReportError",
    "BenchmarkRunResult",
    "BenchmarkRunner",
    "BenchmarkRunnerError",
    "BenchmarkSample",
    "MetricResult",
    "SampleEvaluation",
    "build_report",
    "calculate_metrics",
    "save_report",
]