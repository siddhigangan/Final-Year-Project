"""Hosted OpenAI-compatible generation provider for SecureCodeRAG."""

from __future__ import annotations

import time
from typing import Any

import requests

from src.generation.base import (
    GenerationConfigurationError,
    GenerationError,
    GenerationInputError,
    GenerationProvider,
)
from src.models import GenerationRequest, GenerationResult, GenerationStatus


class HostedOpenAIConnectionError(GenerationError):
    """Raised when the hosted generation endpoint cannot be reached."""


class HostedOpenAIResponseError(GenerationError):
    """Raised when the hosted endpoint returns an invalid response."""


class HostedOpenAIProvider(GenerationProvider):
    """Generate code using an OpenAI-compatible hosted API."""

    DEFAULT_PROVIDER_NAME = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the hosted OpenAI provider."""
        super().__init__(
            model_name=model_name,
            provider_name=self.DEFAULT_PROVIDER_NAME,
            metadata=metadata,
        )

        if not isinstance(api_key, str) or not api_key.strip():
            raise GenerationConfigurationError(
                "api_key must be a non-empty string."
            )

        if not isinstance(base_url, str) or not base_url.strip():
            raise GenerationConfigurationError(
                "base_url must be a non-empty string."
            )

        if timeout_seconds <= 0:
            raise GenerationConfigurationError(
                "timeout_seconds must be greater than zero."
            )

        if temperature < 0:
            raise GenerationConfigurationError(
                "temperature must be greater than or equal to zero."
            )

        if max_tokens is not None and max_tokens <= 0:
            raise GenerationConfigurationError(
                "max_tokens must be greater than zero when provided."
            )

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.max_tokens = max_tokens

    @property
    def supports_streaming(self) -> bool:
        """Return whether the OpenAI-compatible endpoint supports streaming."""
        return True

    @property
    def endpoint(self) -> str:
        """Return the chat-completions endpoint."""
        return f"{self.base_url}/chat/completions"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate code using the hosted OpenAI-compatible endpoint."""
        request = self.prepare_request(request)

        prompt = self._extract_prompt(request)
        payload = self._build_payload(prompt)

        start_time = time.perf_counter()

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise HostedOpenAIConnectionError(
                "Hosted generation request timed out."
            ) from exc
        except requests.ConnectionError as exc:
            raise HostedOpenAIConnectionError(
                f"Unable to connect to {self.base_url!r}."
            ) from exc
        except requests.RequestException as exc:
            raise HostedOpenAIConnectionError(
                f"Hosted generation request failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        response_payload = self._parse_json(response)

        return self._build_generation_result(
            request=request,
            response_payload=response_payload,
            latency_ms=latency_ms,
        )

    def _extract_prompt(self, request: GenerationRequest) -> str:
        """Extract the prepared prompt from request metadata."""
        metadata = getattr(request, "metadata", {})

        if not isinstance(metadata, dict):
            raise GenerationInputError(
                "GenerationRequest.metadata must be a dictionary."
            )

        prompt = metadata.get("prompt")

        if not isinstance(prompt, str) or not prompt.strip():
            raise GenerationInputError(
                "GenerationRequest.metadata['prompt'] must contain a "
                "non-empty prepared prompt."
            )

        return prompt.strip()

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        """Build the OpenAI-compatible chat completion payload."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        return payload

    def _headers(self) -> dict[str, str]:
        """Build HTTP headers without exposing the API key."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_json(response: requests.Response) -> dict[str, Any]:
        """Parse and validate a hosted API response."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise HostedOpenAIResponseError(
                "Hosted endpoint returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise HostedOpenAIResponseError(
                "Hosted endpoint response must be a JSON object."
            )

        error_payload = payload.get("error")

        if error_payload:
            raise HostedOpenAIResponseError(
                f"Hosted endpoint returned an error: {error_payload}"
            )

        return payload

    @staticmethod
    def _build_generation_result(
        *,
        request: GenerationRequest,
        response_payload: dict[str, Any],
        latency_ms: float,
    ) -> GenerationResult:
        """Convert an OpenAI-compatible response to GenerationResult."""
        choices = response_payload.get("choices")

        if not isinstance(choices, list) or not choices:
            raise HostedOpenAIResponseError(
                "Hosted response does not contain a valid choices list."
            )

        first_choice = choices[0]

        if not isinstance(first_choice, dict):
            raise HostedOpenAIResponseError(
                "Hosted response contains an invalid choice."
            )

        message = first_choice.get("message")

        if not isinstance(message, dict):
            raise HostedOpenAIResponseError(
                "Hosted response choice does not contain a valid message."
            )

        generated_code = message.get("content")

        if not isinstance(generated_code, str):
            raise HostedOpenAIResponseError(
                "Hosted response does not contain textual message content."
            )

        generated_code = generated_code.strip()

        if not generated_code:
            raise HostedOpenAIResponseError(
                "Hosted endpoint returned an empty generation."
            )

        usage = response_payload.get("usage")

        if not isinstance(usage, dict):
            usage = {}

        prompt_tokens = HostedOpenAIProvider._extract_optional_int(
            usage,
            "prompt_tokens",
        )

        completion_tokens = HostedOpenAIProvider._extract_optional_int(
            usage,
            "completion_tokens",
        )

        retrieved_chunk_ids = [
            chunk.chunk.chunk_id
            for chunk in request.context
        ]

        metadata = {
            "finish_reason": first_choice.get("finish_reason"),
            "response_id": response_payload.get("id"),
            "response_created": response_payload.get("created"),
            "system_fingerprint": response_payload.get(
                "system_fingerprint"
            ),
        }

        return GenerationResult(
            request_id=request.request_id,
            generated_code=generated_code,
            model_name=request.model_name or "unknown",
            provider_name=request.provider_name or "openai",
            status=GenerationStatus.SUCCESS,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            retrieved_chunk_ids=retrieved_chunk_ids,
            metadata=metadata,
        )

    @staticmethod
    def _extract_optional_int(
        payload: dict[str, Any],
        key: str,
    ) -> int | None:
        """Return an integer field when available."""
        value = payload.get(key)

        if value is None or isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value

        return None

    def get_metadata(self) -> dict[str, Any]:
        """Return provider metadata without exposing the API key."""
        return {
            **super().get_metadata(),
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": bool(self.api_key),
        }