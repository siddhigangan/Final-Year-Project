"""Hugging Face/Sentence Transformers embedding provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProvider,
    normalize_embeddings,
)
from src.models import CodeChunk


class HFCodeEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Hugging Face embedding provider.

        Args:
            model_name: Hugging Face/Sentence Transformers model name.
            device: Torch device such as ``cpu`` or ``cuda``.
            batch_size: Number of texts encoded in one batch.
            normalize: Whether to L2-normalize generated vectors.
            max_seq_length: Optional maximum transformer sequence length.
            model_kwargs: Additional arguments passed to SentenceTransformer.
            encode_kwargs: Additional arguments passed to ``encode``.

        Raises:
            EmbeddingConfigurationError: If configuration is invalid.
        """
        if not model_name or not model_name.strip():
            raise EmbeddingConfigurationError(
                "model_name must be a non-empty string."
            )

        if batch_size <= 0:
            raise EmbeddingConfigurationError(
                "batch_size must be greater than zero."
            )

        if max_seq_length is not None and max_seq_length <= 0:
            raise EmbeddingConfigurationError(
                "max_seq_length must be greater than zero when provided."
            )

        self._model_name = model_name.strip()
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._model_kwargs = dict(model_kwargs or {})
        self._encode_kwargs = dict(encode_kwargs or {})
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the model's embedding dimension."""
        self._ensure_model_loaded()

        if self._dimension is None:
            raise EmbeddingError("Embedding dimension is unavailable.")

        return self._dimension

    @property
    def device(self) -> str | None:
        """Return the configured device."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Return the configured encoding batch size."""
        return self._batch_size

    @property
    def normalize(self) -> bool:
        """Return whether generated embeddings are normalized."""
        return self._normalize

    def load(self) -> None:
        """Load the embedding model into memory."""
        self._ensure_model_loaded()

    def unload(self) -> None:
        """Release the loaded model reference."""
        self._model = None
        self._dimension = None

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Generate embeddings for multiple text values."""
        validated_texts = self._validate_texts(texts)

        if not validated_texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_model_loaded()

        if self._model is None:
            raise EmbeddingError("Embedding model failed to load.")

        kwargs = dict(self._encode_kwargs)
        kwargs.setdefault("batch_size", self._batch_size)
        kwargs.setdefault("convert_to_numpy", True)
        kwargs.setdefault("show_progress_bar", False)

        if self._device is not None:
            kwargs.setdefault("device", self._device)

        try:
            embeddings = self._model.encode(
                validated_texts,
                **kwargs,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Embedding generation failed for model "
                f"{self._model_name!r}: {exc}"
            ) from exc

        array = np.asarray(embeddings, dtype=np.float32)

        if array.ndim == 1:
            array = array.reshape(1, -1)

        validated = self.validate_embeddings(
            array,
            expected_count=len(validated_texts),
        )

        if self._normalize:
            validated = normalize_embeddings(validated)

        return validated

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> np.ndarray:
        """Generate embeddings for SecureCodeRAG code chunks."""
        if chunks is None:
            raise EmbeddingInputError("chunks cannot be None.")

        if not isinstance(chunks, Sequence):
            raise EmbeddingInputError("chunks must be a sequence.")

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

    def metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata for this provider."""
        metadata = super().metadata()
        metadata.update(
            {
                "device": self._device,
                "batch_size": self._batch_size,
                "normalize": self._normalize,
                "max_seq_length": self._max_seq_length,
            }
        )
        return metadata

    def _ensure_model_loaded(self) -> None:
        """Load the transformer model lazily."""
        if self._model is not None:
            return

        try:
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                **self._model_kwargs,
            )

            if self._max_seq_length is not None:
                self._model.max_seq_length = self._max_seq_length

            self._dimension = int(self._model.get_sentence_embedding_dimension())

        except Exception as exc:
            self._model = None
            self._dimension = None

            raise EmbeddingError(
                f"Unable to load embedding model "
                f"{self._model_name!r}: {exc}"
            ) from exc

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        """Validate and normalize text inputs."""
        if texts is None:
            raise EmbeddingInputError("texts cannot be None.")

        if not isinstance(texts, Sequence):
            raise EmbeddingInputError("texts must be a sequence.")

        validated: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingInputError(
                    f"texts[{index}] must be a string."
                )

            if not text.strip():
                raise EmbeddingInputError(
                    f"texts[{index}] cannot be empty."
                )

            validated.append(text)

        return validated