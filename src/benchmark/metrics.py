"""Benchmark metrics for SecureCodeRAG."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import log2


class BenchmarkMetricError(ValueError):
    """Raised when benchmark metric inputs are invalid."""


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Complete benchmark metric result."""

    clean_success_rate: float
    poisoned_success_rate: float
    defended_success_rate: float
    attack_impact: float
    detection_rate: float

    def to_dict(self) -> dict[str, float]:
        """Return metrics as a serializable dictionary."""
        return {
            "clean_success_rate": self.clean_success_rate,
            "poisoned_success_rate": self.poisoned_success_rate,
            "defended_success_rate": self.defended_success_rate,
            "attack_impact": self.attack_impact,
            "detection_rate": self.detection_rate,
        }


def _validate_boolean_sequence(
    values: Iterable[bool],
    name: str,
) -> list[bool]:
    """Validate and materialize a boolean sequence."""
    result = list(values)

    for value in result:
        if not isinstance(value, bool):
            raise BenchmarkMetricError(
                f"{name} must contain only boolean values."
            )

    return result


def _validate_inputs(
    clean_success: Iterable[bool],
    poisoned_success: Iterable[bool],
    defended_success: Iterable[bool],
    detected: Iterable[bool],
) -> tuple[list[bool], list[bool], list[bool], list[bool]]:
    """Validate benchmark metric inputs."""
    clean = _validate_boolean_sequence(clean_success, "clean_success")
    poisoned = _validate_boolean_sequence(
        poisoned_success,
        "poisoned_success",
    )
    defended = _validate_boolean_sequence(
        defended_success,
        "defended_success",
    )
    detection = _validate_boolean_sequence(
        detected,
        "detected",
    )

    lengths = {
        len(clean),
        len(poisoned),
        len(defended),
        len(detection),
    }

    if len(lengths) != 1:
        raise BenchmarkMetricError(
            "All benchmark metric inputs must have the same length."
        )

    if not clean:
        raise BenchmarkMetricError(
            "Benchmark metric inputs cannot be empty."
        )

    return clean, poisoned, defended, detection


def _success_rate(values: Sequence[bool]) -> float:
    """Calculate the proportion of successful samples."""
    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_metrics(
    clean_success: Iterable[bool],
    poisoned_success: Iterable[bool],
    defended_success: Iterable[bool],
    detected: Iterable[bool],
) -> MetricResult:
    """
    Calculate the core SecureCodeRAG benchmark metrics.

    clean_success:
        Whether the clean-code evaluation succeeded.

    poisoned_success:
        Whether the poisoned-code evaluation succeeded.

    defended_success:
        Whether the defended evaluation succeeded.

    detected:
        Whether the poisoning was detected by the defense.
    """
    (
        clean,
        poisoned,
        defended,
        detection,
    ) = _validate_inputs(
        clean_success,
        poisoned_success,
        defended_success,
        detected,
    )

    clean_rate = _success_rate(clean)
    poisoned_rate = _success_rate(poisoned)
    defended_rate = _success_rate(defended)
    detection_rate = _success_rate(detection)

    attack_impact = max(
        0.0,
        clean_rate - poisoned_rate,
    )

    return MetricResult(
        clean_success_rate=clean_rate,
        poisoned_success_rate=poisoned_rate,
        defended_success_rate=defended_rate,
        attack_impact=attack_impact,
        detection_rate=detection_rate,
    )


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _validate_k(k: int) -> None:
    """Validate top-k value."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise BenchmarkMetricError(
            "k must be a positive integer."
        )


def precision_at_k(
    retrieved: Iterable[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Calculate Precision@K."""
    _validate_k(k)

    retrieved_list = list(retrieved)[:k]
    relevant_set = set(relevant)

    if not retrieved_list:
        return 0.0

    hits = sum(
        item in relevant_set
        for item in retrieved_list
    )

    return hits / len(retrieved_list)


def recall_at_k(
    retrieved: Iterable[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Calculate Recall@K."""
    _validate_k(k)

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    retrieved_list = list(retrieved)[:k]

    hits = sum(
        item in relevant_set
        for item in retrieved_list
    )

    return min(
        1.0,
        hits / len(relevant_set),
    )


def hit_rate_at_k(
    retrieved: Iterable[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Calculate Hit Rate@K."""
    _validate_k(k)

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    retrieved_list = list(retrieved)[:k]

    return float(
        any(
            item in relevant_set
            for item in retrieved_list
        )
    )


def reciprocal_rank(
    retrieved: Iterable[str],
    relevant: Iterable[str],
) -> float:
    """Calculate Reciprocal Rank."""
    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    for index, item in enumerate(
        retrieved,
        start=1,
    ):
        if item in relevant_set:
            return 1.0 / index

    return 0.0


def mean_reciprocal_rank(
    rankings: Iterable[Iterable[str]],
    relevant_sets: Iterable[Iterable[str]],
) -> float:
    """Calculate Mean Reciprocal Rank."""
    rankings_list = list(rankings)
    relevant_list = list(relevant_sets)

    if len(rankings_list) != len(relevant_list):
        raise BenchmarkMetricError(
            "rankings and relevant_sets must have the same length."
        )

    if not rankings_list:
        return 0.0

    scores = [
        reciprocal_rank(
            ranking,
            relevant,
        )
        for ranking, relevant in zip(
            rankings_list,
            relevant_list,
            strict=True,
        )
    ]

    return sum(scores) / len(scores)


def ndcg_at_k(
    relevance: Iterable[float],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain."""
    _validate_k(k)

    values = list(relevance)[:k]

    if not values:
        return 0.0

    for value in values:
        if not isinstance(value, (int, float)):
            raise BenchmarkMetricError(
                "relevance values must be numeric."
            )

        if value < 0:
            raise BenchmarkMetricError(
                "relevance values cannot be negative."
            )

    dcg = sum(
        float(value) / log2(index + 2)
        for index, value in enumerate(values)
    )

    ideal = sorted(
        values,
        reverse=True,
    )

    idcg = sum(
        float(value) / log2(index + 2)
        for index, value in enumerate(ideal)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


__all__ = [
    "BenchmarkMetricError",
    "MetricResult",
    "calculate_metrics",
    "hit_rate_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]