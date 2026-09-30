"""Base interfaces for embedding providers used by SecureCodeRAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.models import CodeChunk


class EmbeddingError(RuntimeError):
    """Base exception for embedding-related failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding provider is incorrectly configured."""


class EmbeddingInputError(EmbeddingError):
    """Raised when invalid data is supplied to an embedding provider."""


class EmbeddingProvider(ABC):
    """Abstract interface implemented by all embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured embedding model name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of generated embeddings."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Generate embeddings for a sequence of texts.

        Args:
            texts: Text values to embed.

        Returns:
            A two-dimensional NumPy array with shape
            ``(number_of_texts, embedding_dimension)``.

        Raises:
            EmbeddingInputError: If the input is invalid.
            EmbeddingError: If embedding generation fails.
        """

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text value.

        Args:
            text: Text to embed.

        Returns:
            A one-dimensional NumPy array containing the embedding.

        Raises:
            EmbeddingInputError: If the text is invalid.
            EmbeddingError: If embedding generation fails.
        """
        embeddings = self.embed_texts([text])

        if embeddings.shape[0] != 1:
            raise EmbeddingError(
                "Embedding provider returned an unexpected number of vectors."
            )

        return embeddings[0]

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> np.ndarray:
        """Generate embeddings for code chunks.

        Args:
            chunks: Code chunks to embed.

        Returns:
            A two-dimensional NumPy array containing one vector per chunk.

        Raises:
            EmbeddingInputError: If chunks are invalid.
            EmbeddingError: If embedding generation fails.
        """
        if chunks is None:
            raise EmbeddingInputError("chunks cannot be None.")

        texts: list[str] = []

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, CodeChunk):
                raise EmbeddingInputError(
                    f"chunks[{index}] must be a CodeChunk instance."
                )

            if not chunk.content.strip():
                raise EmbeddingInputError(
                    f"chunks[{index}] contains empty content."
                )

            texts.append(chunk.content)

        return self.embed_texts(texts)

    def validate_embeddings(
        self,
        embeddings: np.ndarray,
        expected_count: int | None = None,
    ) -> np.ndarray:
        """Validate and normalize an embedding matrix.

        Args:
            embeddings: Embedding matrix returned by the provider.
            expected_count: Optional expected number of vectors.

        Returns:
            A validated float32 NumPy array.

        Raises:
            EmbeddingError: If the embeddings have an invalid shape or
                contain invalid numeric values.
        """
        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingError("Embeddings must be returned as a NumPy array.")

        if embeddings.ndim != 2:
            raise EmbeddingError(
                f"Embeddings must be two-dimensional; got ndim={embeddings.ndim}."
            )

        if embeddings.shape[1] != self.dimension:
            raise EmbeddingError(
                "Embedding dimension mismatch: "
                f"expected {self.dimension}, got {embeddings.shape[1]}."
            )

        if expected_count is not None and embeddings.shape[0] != expected_count:
            raise EmbeddingError(
                "Embedding count mismatch: "
                f"expected {expected_count}, got {embeddings.shape[0]}."
            )

        if not np.isfinite(embeddings).all():
            raise EmbeddingError("Embeddings contain NaN or infinite values.")

        return np.asarray(embeddings, dtype=np.float32)

    def validate_vector(self, vector: np.ndarray) -> np.ndarray:
        """Validate a single embedding vector.

        Args:
            vector: One-dimensional embedding vector.

        Returns:
            A validated float32 NumPy array.

        Raises:
            EmbeddingError: If the vector is invalid.
        """
        if not isinstance(vector, np.ndarray):
            raise EmbeddingError("Embedding vector must be a NumPy array.")

        if vector.ndim != 1:
            raise EmbeddingError(
                f"Embedding vector must be one-dimensional; got ndim={vector.ndim}."
            )

        if vector.shape[0] != self.dimension:
            raise EmbeddingError(
                "Embedding dimension mismatch: "
                f"expected {self.dimension}, got {vector.shape[0]}."
            )

        if not np.isfinite(vector).all():
            raise EmbeddingError(
                "Embedding vector contains NaN or infinite values."
            )

        return np.asarray(vector, dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        """Return provider metadata useful for experiment reproducibility."""
        return {
            "provider": self.__class__.__name__,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }

    def __repr__(self) -> str:
        """Return a concise provider representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"dimension={self.dimension})"
        )


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding matrix.

    Args:
        embeddings: Two-dimensional embedding matrix.

    Returns:
        L2-normalized float32 matrix.

    Raises:
        EmbeddingError: If the input is not a valid matrix.
    """
    array = np.asarray(embeddings, dtype=np.float32)

    if array.ndim != 2:
        raise EmbeddingError(
            f"Embedding matrix must be two-dimensional; got ndim={array.ndim}."
        )

    if not np.isfinite(array).all():
        raise EmbeddingError("Embedding matrix contains NaN or infinite values.")

    norms = np.linalg.norm(array, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise EmbeddingError("Cannot normalize zero-length embedding vectors.")

    return array / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a single embedding vector.

    Args:
        vector: One-dimensional embedding vector.

    Returns:
        L2-normalized float32 vector.

    Raises:
        EmbeddingError: If the vector is invalid or has zero norm.
    """
    array = np.asarray(vector, dtype=np.float32)

    if array.ndim != 1:
        raise EmbeddingError(
            f"Embedding vector must be one-dimensional; got ndim={array.ndim}."
        )

    if not np.isfinite(array).all():
        raise EmbeddingError("Embedding vector contains NaN or infinite values.")

    norm = np.linalg.norm(array)

    if norm == 0:
        raise EmbeddingError("Cannot normalize a zero-length embedding vector.")

    return array / norm