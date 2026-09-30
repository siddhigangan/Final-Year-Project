"""Second-stage reranking for SecureCodeRAG retrieval results."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.models import RetrievedChunk


class RerankerError(Exception):
    """Base exception for reranking failures."""


class RerankerConfigurationError(RerankerError):
    """Raised when reranker configuration is invalid."""


class RerankerInputError(RerankerError):
    """Raised when reranker input is invalid."""


@dataclass(frozen=True)
class RerankRequest:
    """Represents a reranking request."""

    query: str
    results: tuple[RetrievedChunk, ...]
    top_k: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise RerankerInputError(
                "query must be a string."
            )

        if not self.query.strip():
            raise RerankerInputError(
                "query cannot be empty."
            )

        if not isinstance(self.results, tuple):
            raise RerankerInputError(
                "results must be a tuple of RetrievedChunk objects."
            )

        if self.top_k is not None:
            if (
                not isinstance(self.top_k, int)
                or isinstance(self.top_k, bool)
            ):
                raise RerankerInputError(
                    "top_k must be an integer or None."
                )

            if self.top_k <= 0:
                raise RerankerInputError(
                    "top_k must be greater than zero."
                )

        for result in self.results:
            if not isinstance(result, RetrievedChunk):
                raise RerankerInputError(
                    "results must contain only RetrievedChunk objects."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reranking request."""
        return {
            "query": self.query,
            "result_count": len(self.results),
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class RerankResult:
    """Represents the output of a reranking operation."""

    results: tuple[RetrievedChunk, ...]
    query: str
    reranker_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_count(self) -> int:
        """Return the number of reranked results."""
        return len(self.results)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return reranked chunk IDs."""
        return tuple(
            result.chunk.chunk_id
            for result in self.results
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reranking result."""
        return {
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "query": self.query,
            "reranker_name": self.reranker_name,
            "result_count": self.result_count,
            "chunk_ids": list(self.chunk_ids),
            "metadata": dict(self.metadata),
        }


class BaseReranker(ABC):
    """Abstract interface for all SecureCodeRAG rerankers."""

    provider_name = "abstract"

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
        top_k: int | None = None,
    ) -> RerankResult:
        """Rerank retrieved results."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the reranker name."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return reranker metadata."""


class LexicalReranker(BaseReranker):
    """Deterministic lexical reranker for retrieved code chunks."""

    provider_name = "lexical"

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*"
    )

    def __init__(
        self,
        name: str = "lexical",
        retrieval_weight: float = 0.35,
        lexical_weight: float = 0.65,
        case_sensitive: bool = False,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise RerankerConfigurationError(
                "name must be a non-empty string."
            )

        if retrieval_weight < 0:
            raise RerankerConfigurationError(
                "retrieval_weight cannot be negative."
            )

        if lexical_weight < 0:
            raise RerankerConfigurationError(
                "lexical_weight cannot be negative."
            )

        total_weight = (
            retrieval_weight
            + lexical_weight
        )

        if total_weight <= 0:
            raise RerankerConfigurationError(
                "At least one reranking weight must be positive."
            )

        self._name = name.strip()
        self._retrieval_weight = (
            retrieval_weight / total_weight
        )
        self._lexical_weight = (
            lexical_weight / total_weight
        )
        self._case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        """Return the reranker name."""
        return self._name

    @property
    def retrieval_weight(self) -> float:
        """Return the normalized retrieval-score weight."""
        return self._retrieval_weight

    @property
    def lexical_weight(self) -> float:
        """Return the normalized lexical-score weight."""
        return self._lexical_weight

    @property
    def metadata(self) -> dict[str, Any]:
        """Return reranker configuration metadata."""
        return {
            "provider": self.provider_name,
            "name": self._name,
            "retrieval_weight": self._retrieval_weight,
            "lexical_weight": self._lexical_weight,
            "case_sensitive": self._case_sensitive,
        }

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
        top_k: int | None = None,
    ) -> RerankResult:
        """Rerank retrieved chunks using lexical relevance."""
        request = self._build_request(
            query=query,
            results=results,
            top_k=top_k,
        )

        query_tokens = self._tokenize(
            request.query
        )

        scored: list[
            tuple[
                float,
                float,
                int,
                RetrievedChunk,
            ]
        ] = []

        for result in request.results:
            lexical_score = self._lexical_score(
                query_tokens,
                result.chunk.content,
            )

            combined_score = (
                self._retrieval_weight
                * self._normalize_retrieval_score(
                    result.retrieval_score
                )
                + self._lexical_weight
                * lexical_score
            )

            scored.append(
                (
                    combined_score,
                    lexical_score,
                    result.rank,
                    result,
                )
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

        if request.top_k is not None:
            scored = scored[:request.top_k]

        reranked: list[RetrievedChunk] = []

        for rank, (
            combined_score,
            lexical_score,
            _original_rank,
            result,
        ) in enumerate(
            scored,
            start=1,
        ):
            retrieval_metadata = dict(
                result.retrieval_metadata
            )

            retrieval_metadata.update(
                {
                    "reranker": self._name,
                    "lexical_score": lexical_score,
                    "combined_rerank_score": combined_score,
                    "original_rank": result.rank,
                }
            )

            reranked.append(
                RetrievedChunk(
                    chunk=result.chunk,
                    retrieval_score=result.retrieval_score,
                    rank=rank,
                    retriever_name=result.retriever_name,
                    rerank_score=combined_score,
                    retrieval_metadata=retrieval_metadata,
                )
            )

        return RerankResult(
            results=tuple(reranked),
            query=request.query,
            reranker_name=self._name,
            metadata={
                **self.metadata,
                "input_count": len(request.results),
                "output_count": len(reranked),
            },
        )

    def _build_request(
        self,
        query: str,
        results: list[RetrievedChunk]
        | tuple[RetrievedChunk, ...],
        top_k: int | None,
    ) -> RerankRequest:
        """Validate and construct a reranking request."""
        if isinstance(results, list):
            normalized_results = tuple(results)
        elif isinstance(results, tuple):
            normalized_results = results
        else:
            raise RerankerInputError(
                "results must be a list or tuple."
            )

        return RerankRequest(
            query=query,
            results=normalized_results,
            top_k=top_k,
        )

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        """Tokenize query/code text deterministically."""
        if not self._case_sensitive:
            text = text.lower()

        return self.TOKEN_PATTERN.findall(
            text
        )

    def _lexical_score(
        self,
        query_tokens: list[str],
        content: str,
    ) -> float:
        """Calculate normalized query-token overlap."""
        if not query_tokens:
            return 0.0

        content_tokens = self._tokenize(
            content
        )

        if not content_tokens:
            return 0.0

        query_counter = Counter(
            query_tokens
        )
        content_counter = Counter(
            content_tokens
        )

        matched = 0

        for token, query_count in query_counter.items():
            matched += min(
                query_count,
                content_counter.get(token, 0),
            )

        total_query_tokens = sum(
            query_counter.values()
        )

        if total_query_tokens == 0:
            return 0.0

        return matched / total_query_tokens

    @staticmethod
    def _normalize_retrieval_score(
        score: float,
    ) -> float:
        """Normalize a similarity score to the [0, 1] interval."""
        if score <= 0:
            return 0.0

        if score >= 1:
            return 1.0

        return float(score)

    def __repr__(self) -> str:
        """Return a useful developer representation."""
        return (
            "LexicalReranker("
            f"name={self._name!r}, "
            f"retrieval_weight={self._retrieval_weight:.3f}, "
            f"lexical_weight={self._lexical_weight:.3f}"
            ")"
        )