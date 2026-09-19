"""Generation-provider factory for SecureCodeRAG."""

from __future__ import annotations

import os
from typing import Any

from src.config import AppConfig
from src.generation.base import (
    GenerationConfigurationError,
    GenerationProvider,
)
from src.generation.hosted_openai import HostedOpenAIProvider
from src.generation.local_ollama import LocalOllamaProvider


class GenerationFactory:
    """Create configured generation providers."""

    SUPPORTED_PROVIDERS = frozenset({"ollama", "openai"})

    @classmethod
    def create(
        cls,
        config: AppConfig,
        *,
        provider_name: str | None = None,
        **overrides: Any,
    ) -> GenerationProvider:
        """Create a generation provider from application configuration.

        Args:
            config: SecureCodeRAG application configuration.
            provider_name: Optional provider override.
            **overrides: Optional provider-specific configuration overrides.

        Returns:
            A configured ``GenerationProvider``.

        Raises:
            GenerationConfigurationError: If the provider is unsupported
                or required configuration is missing.
        """
        if not isinstance(config, AppConfig):
            raise GenerationConfigurationError(
                "config must be an AppConfig instance."
            )

        selected_provider = (
            provider_name or config.code_llm_provider
        ).strip().lower()

        if selected_provider not in cls.SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(cls.SUPPORTED_PROVIDERS))
            raise GenerationConfigurationError(
                f"Unsupported generation provider "
                f"{selected_provider!r}. Supported providers: {supported}."
            )

        if selected_provider == "ollama":
            return cls._create_ollama(config, overrides)

        return cls._create_openai(config, overrides)

    @classmethod
    def _create_ollama(
        cls,
        config: AppConfig,
        overrides: dict[str, Any],
    ) -> LocalOllamaProvider:
        """Create the configured local Ollama provider."""
        model_name = overrides.get("model_name", config.code_llm)
        base_url = overrides.get(
            "base_url",
            config.code_llm_base_url,
        )

        timeout_seconds = overrides.get(
            "timeout_seconds",
            config.extra.get(
                "generation_timeout_seconds",
                120.0,
            ),
        )

        temperature = overrides.get(
            "temperature",
            config.extra.get(
                "generation_temperature",
                0.0,
            ),
        )

        max_tokens = overrides.get(
            "max_tokens",
            config.extra.get("generation_max_tokens"),
        )

        keep_alive = overrides.get(
            "keep_alive",
            config.extra.get("ollama_keep_alive"),
        )

        return LocalOllamaProvider(
            model_name=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
            keep_alive=keep_alive,
        )

    @classmethod
    def _create_openai(
        cls,
        config: AppConfig,
        overrides: dict[str, Any],
    ) -> HostedOpenAIProvider:
        """Create the configured hosted OpenAI provider."""
        model_name = overrides.get(
            "model_name",
            config.hosted_llm_model,
        )

        base_url = overrides.get(
            "base_url",
            os.getenv(
                "OPENAI_BASE_URL",
                "https://api.openai.com/v1",
            ),
        )

        api_key = overrides.get(
            "api_key",
            os.getenv("OPENAI_API_KEY", ""),
        )

        timeout_seconds = overrides.get(
            "timeout_seconds",
            config.extra.get(
                "generation_timeout_seconds",
                120.0,
            ),
        )

        temperature = overrides.get(
            "temperature",
            config.extra.get(
                "generation_temperature",
                0.0,
            ),
        )

        max_tokens = overrides.get(
            "max_tokens",
            config.extra.get("generation_max_tokens"),
        )

        if not api_key:
            raise GenerationConfigurationError(
                "OPENAI_API_KEY is required when using the OpenAI "
                "generation provider."
            )

        return HostedOpenAIProvider(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @classmethod
    def supported_providers(cls) -> tuple[str, ...]:
        """Return supported provider names in deterministic order."""
        return tuple(sorted(cls.SUPPORTED_PROVIDERS))