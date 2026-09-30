"""Factory for constructing SecureCodeRAG embedding providers."""

from __future__ import annotations

from typing import Any

from src.config import AppConfig
from src.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
)
from src.embeddings.hf_code_embeddings import HFCodeEmbeddingProvider


class EmbeddingFactoryError(EmbeddingConfigurationError):
    """Raised when an embedding provider cannot be constructed."""


class EmbeddingFactory:
    """Construct embedding providers from application configuration."""

    SUPPORTED_PROVIDERS = frozenset(
        {
            "huggingface",
            "hf",
            "sentence-transformers",
            "sentence_transformer",
        }
    )

    @classmethod
    def create(
        cls,
        config: AppConfig,
        *,
        provider_name: str | None = None,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        max_seq_length: int | None = None,
    ) -> EmbeddingProvider:
        """Create an embedding provider from application configuration.

        Args:
            config: Loaded SecureCodeRAG application configuration.
            provider_name: Optional provider override.
            model_name: Optional model-name override.
            device: Optional device override.
            batch_size: Optional batch-size override.
            normalize: Optional normalization override.
            max_seq_length: Optional sequence-length override.

        Returns:
            A configured embedding provider.

        Raises:
            EmbeddingFactoryError: If configuration is invalid or the
                provider is unsupported.
        """
        if not isinstance(config, AppConfig):
            raise EmbeddingFactoryError(
                "config must be an AppConfig instance."
            )

        selected_provider = (
            provider_name
            if provider_name is not None
            else config.embedding_provider
        )

        if not selected_provider:
            raise EmbeddingFactoryError(
                "embedding_provider must be configured."
            )

        selected_provider = str(selected_provider).strip().lower()

        if selected_provider not in cls.SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(cls.SUPPORTED_PROVIDERS))

            raise EmbeddingFactoryError(
                f"Unsupported embedding provider {selected_provider!r}. "
                f"Supported providers: {supported}."
            )

        selected_model = (
            model_name
            if model_name is not None
            else config.embedding_model
        )

        if selected_model is None or not str(selected_model).strip():
            raise EmbeddingFactoryError(
                "embedding_model must be configured."
            )

        selected_batch_size = 32 if batch_size is None else batch_size

        try:
            selected_batch_size = int(selected_batch_size)
        except (TypeError, ValueError) as exc:
            raise EmbeddingFactoryError(
                "batch_size must be an integer."
            ) from exc

        if selected_batch_size <= 0:
            raise EmbeddingFactoryError(
                "batch_size must be greater than zero."
            )

        if normalize is None:
            selected_normalize = True
        else:
            selected_normalize = bool(normalize)

        selected_max_seq_length = max_seq_length

        if selected_max_seq_length is not None:
            try:
                selected_max_seq_length = int(selected_max_seq_length)
            except (TypeError, ValueError) as exc:
                raise EmbeddingFactoryError(
                    "max_seq_length must be an integer."
                ) from exc

            if selected_max_seq_length <= 0:
                raise EmbeddingFactoryError(
                    "max_seq_length must be greater than zero."
                )

        return HFCodeEmbeddingProvider(
            model_name=str(selected_model).strip(),
            device=device,
            batch_size=selected_batch_size,
            normalize=selected_normalize,
            max_seq_length=selected_max_seq_length,
        )

    @classmethod
    def create_from_values(
        cls,
        *,
        provider_name: str,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: int | None = None,
    ) -> EmbeddingProvider:
        """Create a provider directly from explicit configuration values.

        This helper is useful for tests and isolated experiments where a
        complete AppConfig is unnecessary.
        """
        if not provider_name or not provider_name.strip():
            raise EmbeddingFactoryError(
                "provider_name must be non-empty."
            )

        if not model_name or not model_name.strip():
            raise EmbeddingFactoryError(
                "model_name must be non-empty."
            )

        normalized_provider = provider_name.strip().lower()

        if normalized_provider not in cls.SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(cls.SUPPORTED_PROVIDERS))

            raise EmbeddingFactoryError(
                f"Unsupported embedding provider {normalized_provider!r}. "
                f"Supported providers: {supported}."
            )

        if batch_size <= 0:
            raise EmbeddingFactoryError(
                "batch_size must be greater than zero."
            )

        if max_seq_length is not None and max_seq_length <= 0:
            raise EmbeddingFactoryError(
                "max_seq_length must be greater than zero."
            )

        return HFCodeEmbeddingProvider(
            model_name=model_name.strip(),
            device=device,
            batch_size=batch_size,
            normalize=normalize,
            max_seq_length=max_seq_length,
        )

    @staticmethod
    def provider_metadata(provider: EmbeddingProvider) -> dict[str, Any]:
        """Return metadata describing an embedding provider."""
        if not isinstance(provider, EmbeddingProvider):
            raise EmbeddingFactoryError(
                "provider must implement EmbeddingProvider."
            )

        return provider.metadata()