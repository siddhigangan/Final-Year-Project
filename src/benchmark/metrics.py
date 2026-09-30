"""Benchmark metrics for SecureCodeRAG poisoning experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class BenchmarkMetricError(ValueError):
    """Raised when benchmark metric inputs are invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Aggregate success/detection rates for one benchmark run."""

    clean_success_rate: float
    poisoned_success_rate: float
    defended_success_rate: float
    attack_impact: float
    detection_rate: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a serializable representation."""
        return {
            "clean_success_rate": self.clean_success_rate,
            "poisoned_success_rate": self.poisoned_success_rate,
            "defended_success_rate": self.defended_success_rate,
            "attack_impact": self.attack_impact,
            "detection_rate": self.detection_rate,
            "sample_count": self.sample_count,
        }


def _validate_bool_sequence(name: str, values: Sequence[bool]) -> list[bool]:
    if not isinstance(values, (list, tuple)):
        raise BenchmarkMetricError(f"{name} must be a list or tuple of bool values.")

    result = list(values)

    for item in result:
        if not isinstance(item, bool):
            raise BenchmarkMetricError(f"{name} must contain only bool values.")

    return result


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_metrics(
    clean_success: Sequence[bool],
    poisoned_success: Sequence[bool],
    defended_success: Sequence[bool],
    detected: Sequence[bool],
) -> BenchmarkMetrics:
    """Calculate aggregate benchmark metrics for a poisoning experiment."""

    clean_list = _validate_bool_sequence("clean_success", clean_success)
    poisoned_list = _validate_bool_sequence("poisoned_success", poisoned_success)
    defended_list = _validate_bool_sequence("defended_success", defended_success)
    detected_list = _validate_bool_sequence("detected", detected)

    lengths = {
        len(clean_list),
        len(poisoned_list),
        len(defended_list),
        len(detected_list),
    }

    if len(lengths) != 1:
        raise BenchmarkMetricError(
            "clean_success, poisoned_success, defended_success, and "
            "detected must all have the same length."
        )

    if not clean_list:
        raise BenchmarkMetricError(
            "Cannot calculate metrics for an empty benchmark run."
        )

    clean_rate = _rate(clean_list)
    poisoned_rate = _rate(poisoned_list)
    defended_rate = _rate(defended_list)
    detection_rate = _rate(detected_list)

    return BenchmarkMetrics(
        clean_success_rate=clean_rate,
        poisoned_success_rate=poisoned_rate,
        defended_success_rate=defended_rate,
        attack_impact=clean_rate - poisoned_rate,
        detection_rate=detection_rate,
        sample_count=len(clean_list),
    )


__all__ = [
    "BenchmarkMetricError",
    "BenchmarkMetrics",
    "calculate_metrics",
]
