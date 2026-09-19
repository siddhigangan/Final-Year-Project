"""Query retrieval orchestration for SecureCodeRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.embeddings.base import EmbeddingProvider
from src.models import CodeChunk, RetrievedChunk
from src.vectorstore.base import VectorStore


class RetrieverError(Exception):
    """Base exception for retrieval failures."""


class RetrieverConfigurationError(RetrieverError):
    """Raised when the retriever configuration is invalid."""


class RetrieverInputError(RetrieverError):
    """Raised when retrieval input is invalid."""


@dataclass(frozen=True)
class RetrievalRequest:
    """Represents a single retrieval request."""

    query: str
    top_k: int = 5
    metadata_filter: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise RetrieverInputError(
                "query must be a string."
            )

        if not self.query.strip():
            raise RetrieverInputError(
                "query cannot be empty."
            )

        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
        ):
            raise RetrieverInputError(
                "top_k must be an integer."
            )

        if self.top_k <= 0:
            raise RetrieverInputError(
                "top_k must be greater than zero."
            )

        if not isinstance(self.metadata_filter, dict):
            raise RetrieverInputError(
                "metadata_filter must be a dictionary."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retrieval request."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "metadata_filter": dict(self.metadata_filter),
        }


@dataclass(frozen=True)
class RetrievalResponse:
    """Represents the result of a retrieval operation."""

    query: str
    results: tuple[RetrievedChunk, ...]
    requested_top_k: int
    retriever_name: str
    embedding_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_count(self) -> int:
        """Return the number of retrieved results."""
        return len(self.results)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return retrieved chunk IDs in ranking order."""
        return tuple(
            result.chunk.chunk_id
            for result in self.results
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the retrieval response."""
        return {
            "query": self.query,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "requested_top_k": self.requested_top_k,
            "result_count": self.result_count,
            "retriever_name": self.retriever_name,
            "embedding_model": self.embedding_model,
            "metadata": dict(self.metadata),
        }


class Retriever:
    """Orchestrates query embedding and vector-store retrieval."""

    DEFAULT_NAME = "default"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        name: str = DEFAULT_NAME,
    ) -> None:
        if not isinstance(
            embedding_provider,
            EmbeddingProvider,
        ):
            raise RetrieverConfigurationError(
                "embedding_provider must implement EmbeddingProvider."
            )

        if not isinstance(vector_store, VectorStore):
            raise RetrieverConfigurationError(
                "vector_store must implement VectorStore."
            )

        if not isinstance(name, str) or not name.strip():
            raise RetrieverConfigurationError(
                "name must be a non-empty string."
            )

        if (
            embedding_provider.dimension is not None
            and embedding_provider.dimension
            != vector_store.dimension
        ):
            raise RetrieverConfigurationError(
                "Embedding dimension does not match vector-store "
                f"dimension: {embedding_provider.dimension} != "
                f"{vector_store.dimension}"
            )

        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._name = name.strip()

    @property
    def name(self) -> str:
        """Return the retriever name."""
        return self._name

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the configured embedding provider."""
        return self._embedding_provider

    @property
    def vector_store(self) -> VectorStore:
        """Return the configured vector store."""
        return self._vector_store

    @property
    def embedding_model(self) -> str:
        """Return the embedding model name."""
        return self._embedding_provider.model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._vector_store.dimension

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Retrieve the most relevant chunks for a query."""
        request = RetrievalRequest(
            query=query,
            top_k=top_k,
            metadata_filter=(
                {}
                if metadata_filter is None
                else metadata_filter
            ),
        )

        query_embedding = self._embed_query(
            request.query
        )

        results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )

        filtered_results = self._apply_metadata_filter(
            results,
            request.metadata_filter,
        )

        normalized_results = self._normalize_results(
            filtered_results
        )

        return RetrievalResponse(
            query=request.query,
            results=tuple(normalized_results),
            requested_top_k=request.top_k,
            retriever_name=self._name,
            embedding_model=self.embedding_model,
            metadata={
                "vector_store": self._vector_store.provider,
                "embedding_dimension": self.dimension,
                "metadata_filter": dict(
                    request.metadata_filter
                ),
            },
        )

    def retrieve_request(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """Retrieve chunks using a RetrievalRequest."""
        if not isinstance(
            request,
            RetrievalRequest,
        ):
            raise RetrieverInputError(
                "request must be a RetrievalRequest."
            )

        return self.retrieve(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter,
        )

    def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return only the retrieved chunks."""
        response = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return list(response.results)

    def retrieve_codes(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[CodeChunk]:
        """Return only the underlying CodeChunk objects."""
        response = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return [
            result.chunk
            for result in response.results
        ]

    def _embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Embed and validate a retrieval query."""
        try:
            embedding = self._embedding_provider.embed_text(
                query
            )
        except Exception as exc:
            raise RetrieverError(
                f"Failed to embed retrieval query: {exc}"
            ) from exc

        try:
            validated = self._embedding_provider.validate_vector(
                embedding
            )
        except Exception as exc:
            raise RetrieverError(
                f"Embedding provider returned an invalid query "
                f"vector: {exc}"
            ) from exc

        return np.asarray(
            validated,
            dtype=np.float32,
        )

    def _apply_metadata_filter(
        self,
        results: list[RetrievedChunk],
        metadata_filter: dict[str, Any],
    ) -> list[RetrievedChunk]:
        """Filter retrieval results using chunk metadata."""
        if not metadata_filter:
            return results

        filtered: list[RetrievedChunk] = []

        for result in results:
            if self._matches_metadata_filter(
                result.chunk,
                metadata_filter,
            ):
                filtered.append(result)

        return filtered

    @staticmethod
    def _matches_metadata_filter(
        chunk: CodeChunk,
        metadata_filter: dict[str, Any],
    ) -> bool:
        """Return whether a chunk satisfies all filters."""
        chunk_metadata = {
            "chunk_id": chunk.chunk_id,
            "source_file_id": chunk.source_file_id,
            "repository_id": chunk.repository_id,
            "language": chunk.language.value,
            "relative_path": chunk.metadata.get(
                "relative_path"
            ),
            "trust": chunk.trust.value,
            "is_documentation": chunk.is_documentation,
            "symbol_name": chunk.symbol_name,
            "symbol_type": chunk.symbol_type,
        }

        chunk_metadata.update(chunk.metadata)

        for key, expected_value in metadata_filter.items():
            actual_value = chunk_metadata.get(key)

            if isinstance(expected_value, (set, frozenset, list, tuple)):
                if actual_value not in expected_value:
                    return False
                continue

            if actual_value != expected_value:
                return False

        return True

    def _normalize_results(
        self,
        results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Normalize ranking and retriever metadata."""
        normalized: list[RetrievedChunk] = []

        for rank, result in enumerate(
            results,
            start=1,
        ):
            metadata = dict(
                result.retrieval_metadata
            )

            metadata.update(
                {
                    "retriever": self._name,
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": self.dimension,
                }
            )

            normalized.append(
                RetrievedChunk(
                    chunk=result.chunk,
                    retrieval_score=result.retrieval_score,
                    rank=rank,
                    retriever_name=self._name,
                    rerank_score=result.rerank_score,
                    retrieval_metadata=metadata,
                )
            )

        return normalized

    def metadata(self) -> dict[str, Any]:
        """Return retriever configuration metadata."""
        return {
            "name": self._name,
            "embedding_provider": (
                self._embedding_provider.__class__.__name__
            ),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.dimension,
            "vector_store": self._vector_store.provider,
            "vector_store_size": self._vector_store.size,
        }

    def __repr__(self) -> str:
        """Return a useful developer representation."""
        return (
            "Retriever("
            f"name={self._name!r}, "
            f"embedding_model={self.embedding_model!r}, "
            f"dimension={self.dimension}, "
            f"vector_store={self._vector_store.provider!r}"
            ")"
        )