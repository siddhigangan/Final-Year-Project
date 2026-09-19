from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import log2
from typing import Any

from src.models import RetrievedChunk


class RetrievalMetricsError(Exception):
    """Base exception for retrieval metric errors."""


class RetrievalMetricsInputError(RetrievalMetricsError, ValueError):
    """Raised when retrieval metric inputs are invalid."""


@dataclass(frozen=True)
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    k: int
    retrieved_count: int
    relevant_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return metrics as a serializable dictionary."""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
            "k": self.k,
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "metadata": dict(self.metadata),
        }


def _validate_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool):
        raise RetrievalMetricsInputError("k must be an integer.")

    if k <= 0:
        raise RetrievalMetricsInputError("k must be greater than zero.")

    return k


def _validate_retrieved(
    retrieved: Iterable[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Validate and materialize retrieved chunks."""
    if retrieved is None:
        raise RetrievalMetricsInputError("retrieved cannot be None.")

    items = list(retrieved)

    if any(not isinstance(item, RetrievedChunk) for item in items):
        raise RetrievalMetricsInputError(
            "retrieved must contain only RetrievedChunk instances."
        )

    return items


def _validate_relevant_ids(
    relevant_ids: Iterable[str],
) -> set[str]:
    """Validate and normalize relevant chunk identifiers."""
    if relevant_ids is None:
        raise RetrievalMetricsInputError("relevant_ids cannot be None.")

    values = set(relevant_ids)

    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise RetrievalMetricsInputError(
            "relevant_ids must contain non-empty strings."
        )

    return values


def _top_k(
    retrieved: Sequence[RetrievedChunk],
    k: int,
) -> list[RetrievedChunk]:
    return list(retrieved[:k])


def _is_relevant(
    chunk: RetrievedChunk,
    relevant_ids: set[str],
) -> bool:
    return chunk.chunk.chunk_id in relevant_ids


def precision_at_k(
    retrieved: Iterable[RetrievedChunk],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision@K is the fraction of retrieved top-k chunks that are relevant.
    """
    items = _validate_retrieved(retrieved)
    relevant = _validate_relevant_ids(relevant_ids)
    k = _validate_k(k)

    top_k = _top_k(items, k)

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for item in top_k if _is_relevant(item, relevant)
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved: Iterable[RetrievedChunk],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K is the fraction of all relevant chunks that appear in top-k.
    """
    items = _validate_retrieved(retrieved)
    relevant = _validate_relevant_ids(relevant_ids)
    k = _validate_k(k)

    if not relevant:
        return 0.0

    top_k = _top_k(items, k)

    retrieved_relevant = {
        item.chunk.chunk_id
        for item in top_k
        if _is_relevant(item, relevant)
    }

    return len(retrieved_relevant) / len(relevant)


def hit_rate_at_k(
    retrieved: Iterable[RetrievedChunk],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate Hit@K.

    Returns 1.0 when at least one relevant chunk occurs in top-k.
    """
    items = _validate_retrieved(retrieved)
    relevant = _validate_relevant_ids(relevant_ids)
    k = _validate_k(k)

    return float(
        any(
            _is_relevant(item, relevant)
            for item in _top_k(items, k)
        )
    )


def reciprocal_rank(
    retrieved: Iterable[RetrievedChunk],
    relevant_ids: Iterable[str],
) -> float:
    """
    Calculate reciprocal rank.

    Returns 1/rank of the first relevant retrieved chunk.
    """
    items = _validate_retrieved(retrieved)
    relevant = _validate_relevant_ids(relevant_ids)

    for rank, item in enumerate(items, start=1):
        if _is_relevant(item, relevant):
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: Iterable[RetrievedChunk],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """
    Calculate binary-relevance NDCG@K.
    """
    items = _validate_retrieved(retrieved)
    relevant = _validate_relevant_ids(relevant_ids)
    k = _validate_k(k)

    if not relevant:
        return 0.0

    top_k = _top_k(items, k)

    dcg = 0.0

    for rank, item in enumerate(top_k, start=1):
        if _is_relevant(item, relevant):
            dcg += 1.0 / log2(rank + 1)

    ideal_count = min(k, len(relevant))

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def calculate_retrieval_metrics(
    retrieved: Iterable[RetrievedChunk],
    relevant_ids: Iterable[str],
    k: int,
    metadata: dict[str, Any] | None = None,
) -> RetrievalMetrics:
    """Calculate the complete retrieval metric set."""
    items = _validate_retrieved(retrieved)
    relevant = _validate_relevant_ids(relevant_ids)
    k = _validate_k(k)

    return RetrievalMetrics(
        precision_at_k=precision_at_k(items, relevant, k),
        recall_at_k=recall_at_k(items, relevant, k),
        hit_rate_at_k=hit_rate_at_k(items, relevant, k),
        reciprocal_rank=reciprocal_rank(items, relevant),
        ndcg_at_k=ndcg_at_k(items, relevant, k),
        k=k,
        retrieved_count=len(items),
        relevant_count=len(relevant),
        metadata=dict(metadata or {}),
    )