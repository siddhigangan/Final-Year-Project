"""Provider-agnostic abstractions for SecureCodeRAG code generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models import GenerationRequest, GenerationResult


class GenerationError(RuntimeError):
    """Base exception for generation-related failures."""


class GenerationConfigurationError(GenerationError):
    """Raised when a generation provider is incorrectly configured."""


class GenerationInputError(GenerationError):
    """Raised when a generation request is invalid."""


class GenerationProvider(ABC):
    """Abstract interface implemented by all code-generation providers."""

    provider_name: str = "unknown"
    model_name: str = "unknown"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a generation provider.

        Args:
            model_name: Identifier of the model used by the provider.
            provider_name: Optional provider identifier. When omitted, the
                class-level provider name is used.
            metadata: Optional provider-specific metadata.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise GenerationConfigurationError(
                "model_name must be a non-empty string."
            )

        resolved_provider_name = provider_name or self.provider_name

        if (
            not isinstance(resolved_provider_name, str)
            or not resolved_provider_name.strip()
        ):
            raise GenerationConfigurationError(
                "provider_name must be a non-empty string."
            )

        self.model_name = model_name.strip()
        self.provider_name = resolved_provider_name.strip()
        self.metadata = dict(metadata or {})

    @property
    def name(self) -> str:
        """Return the stable provider identifier."""
        return self.provider_name

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self.model_name

    @property
    def supports_streaming(self) -> bool:
        """Return whether the provider supports streaming generation.

        Providers that support streaming should override this property.
        The base implementation deliberately defaults to ``False`` so that
        callers do not assume streaming support.
        """
        return False

    def validate_request(self, request: GenerationRequest) -> None:
        """Validate a generation request before sending it to a provider.

        The validation intentionally stays provider-agnostic. Provider-
        specific constraints such as token limits, API parameters, or
        endpoint requirements belong in the concrete provider.
        """
        if not isinstance(request, GenerationRequest):
            raise GenerationInputError(
                "request must be an instance of GenerationRequest."
            )

        query = getattr(request, "query", None)

        if not isinstance(query, str) or not query.strip():
            raise GenerationInputError(
                "GenerationRequest.query must be a non-empty string."
            )

        task = getattr(request, "task", None)

        if not isinstance(task, str) or not task.strip():
            raise GenerationInputError(
                "GenerationRequest.task must be a non-empty string."
            )

        context = getattr(request, "context", None)

        if context is None:
            raise GenerationInputError(
                "GenerationRequest.context must not be None."
            )

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        """Validate and return a generation request.

        Concrete providers may override this method when provider-specific
        normalization is required, while preserving the common validation
        contract.
        """
        self.validate_request(request)
        return request

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate code from a validated request.

        Concrete implementations must perform the actual model invocation
        and return a ``GenerationResult``.
        """

    def get_metadata(self) -> dict[str, Any]:
        """Return non-sensitive provider metadata.

        Concrete providers can extend this metadata with useful diagnostic
        information. Secrets such as API keys must never be returned.
        """
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "supports_streaming": self.supports_streaming,
            **self.metadata,
        }

    def __repr__(self) -> str:
        """Return a concise provider representation."""
        return (
            f"{self.__class__.__name__}("
            f"provider_name={self.provider_name!r}, "
            f"model_name={self.model_name!r})"
        )