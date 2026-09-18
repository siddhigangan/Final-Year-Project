"""Abstract vector-store interface for SecureCodeRAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models import CodeChunk, RetrievedChunk


class VectorStoreError(Exception):
    """Base exception for vector-store failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector-store configuration is invalid."""


class VectorStoreInputError(VectorStoreError):
    """Raised when vector-store input data is invalid."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when requested vector-store data does not exist."""


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Internal vector-search result before model conversion."""

    chunk_id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.chunk_id.strip():
            raise VectorStoreInputError("chunk_id must be a non-empty string.")

        if not np.isfinite(self.score):
            raise VectorStoreInputError("score must be finite.")

        if self.rank < 1:
            raise VectorStoreInputError("rank must be greater than or equal to 1.")

        if not isinstance(self.metadata, dict):
            raise VectorStoreInputError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "metadata": dict(self.metadata),
        }


class VectorStore(ABC):
    """Abstract interface implemented by concrete vector stores."""

    provider_name = "abstract"

    def __init__(self, dimension: int) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise VectorStoreConfigurationError(
                "dimension must be an integer."
            )

        if dimension <= 0:
            raise VectorStoreConfigurationError(
                "dimension must be greater than zero."
            )

        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        return self._dimension

    @property
    def provider(self) -> str:
        """Return the vector-store provider name."""
        return self.provider_name

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed vectors."""

    @abstractmethod
    def add(
        self,
        embeddings: np.ndarray,
        chunks: Sequence[CodeChunk],
    ) -> list[str]:
        """Add embeddings and their corresponding chunks."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search the store using a query embedding."""

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Delete vectors associated with chunk IDs."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors and metadata."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the vector store to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""

    def add_chunks(
        self,
        embeddings: np.ndarray,
        chunks: Sequence[CodeChunk],
    ) -> list[str]:
        """Validate and add embeddings associated with code chunks."""
        embeddings_array = self.validate_embeddings(embeddings)

        if not chunks:
            raise VectorStoreInputError("chunks cannot be empty.")

        if len(embeddings_array) != len(chunks):
            raise VectorStoreInputError(
                "The number of embeddings must match the number of chunks."
            )

        self._validate_chunk_ids(chunks)

        return self.add(embeddings_array, chunks)

    def search_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Validate a query vector and perform similarity search."""
        query = self.validate_query_embedding(query_embedding)

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise VectorStoreInputError("top_k must be an integer.")

        if top_k <= 0:
            raise VectorStoreInputError(
                "top_k must be greater than zero."
            )

        return self.search(query, top_k=top_k)

    def validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Validate and normalize the shape/dtype of multiple embeddings."""
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise VectorStoreInputError(
                "embeddings must be a two-dimensional array."
            )

        if embeddings.shape[1] != self.dimension:
            raise VectorStoreInputError(
                f"Expected embeddings with dimension {self.dimension}, "
                f"received {embeddings.shape[1]}."
            )

        if embeddings.shape[0] == 0:
            raise VectorStoreInputError("embeddings cannot be empty.")

        if not np.issubdtype(embeddings.dtype, np.number):
            raise VectorStoreInputError(
                "embeddings must contain numeric values."
            )

        result = np.asarray(embeddings, dtype=np.float32)

        if not np.all(np.isfinite(result)):
            raise VectorStoreInputError(
                "embeddings must contain only finite values."
            )

        return np.ascontiguousarray(result)

    def validate_query_embedding(self, query_embedding: np.ndarray) -> np.ndarray:
        """Validate a single query embedding."""
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.asarray(query_embedding)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise VectorStoreInputError(
                "query_embedding must contain exactly one vector."
            )

        validated = self.validate_embeddings(query_embedding)

        return validated[0]

    @staticmethod
    def _validate_chunk_ids(chunks: Sequence[CodeChunk]) -> None:
        """Validate chunk IDs and reject duplicates."""
        chunk_ids: list[str] = []

        for chunk in chunks:
            if not isinstance(chunk, CodeChunk):
                raise VectorStoreInputError(
                    "chunks must contain only CodeChunk instances."
                )

            if not chunk.chunk_id.strip():
                raise VectorStoreInputError(
                    "Every chunk must have a non-empty chunk_id."
                )

            chunk_ids.append(chunk.chunk_id)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise VectorStoreInputError(
                "Duplicate chunk IDs are not allowed in one add operation."
            )

    def metadata(self) -> dict[str, Any]:
        """Return common vector-store metadata."""
        return {
            "provider": self.provider,
            "dimension": self.dimension,
            "size": self.size,
        }

    def __len__(self) -> int:
        """Return the number of indexed vectors."""
        return self.size

    def __repr__(self) -> str:
        """Return a concise vector-store representation."""
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider!r}, "
            f"dimension={self.dimension}, "
            f"size={self.size})"
        )