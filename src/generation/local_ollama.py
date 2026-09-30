"""Local Ollama generation provider for SecureCodeRAG."""

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


class OllamaConnectionError(GenerationError):
    """Raised when the Ollama server cannot be reached."""


class OllamaResponseError(GenerationError):
    """Raised when Ollama returns an invalid or unsuccessful response."""


class LocalOllamaProvider(GenerationProvider):
    """Generate code using a locally hosted Ollama model."""

    DEFAULT_PROVIDER_NAME = "ollama"
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        keep_alive: str | int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the local Ollama provider."""
        super().__init__(
            model_name=model_name,
            provider_name=self.DEFAULT_PROVIDER_NAME,
            metadata=metadata,
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

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.keep_alive = keep_alive

    @property
    def supports_streaming(self) -> bool:
        """Return whether Ollama supports streaming generation."""
        return True

    @property
    def endpoint(self) -> str:
        """Return the Ollama generation endpoint."""
        return f"{self.base_url}/api/generate"

    def health_check(self) -> bool:
        """Check whether the Ollama server is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Unable to reach Ollama at {self.base_url!r}."
            ) from exc

        return True

    def list_models(self) -> list[str]:
        """Return models currently available in the local Ollama server."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Unable to query Ollama at {self.base_url!r}."
            ) from exc

        payload = self._parse_json(response)

        models = payload.get("models", [])

        if not isinstance(models, list):
            raise OllamaResponseError(
                "Ollama /api/tags returned an invalid 'models' field."
            )

        names: list[str] = []

        for model in models:
            if not isinstance(model, dict):
                continue

            name = model.get("name")

            if isinstance(name, str) and name.strip():
                names.append(name.strip())

        return names

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate code using the configured Ollama model."""
        request = self.prepare_request(request)

        prompt = self._extract_prompt(request)
        payload = self._build_payload(prompt)

        start_time = time.perf_counter()

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise OllamaConnectionError(
                "Ollama generation request timed out."
            ) from exc
        except requests.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Unable to connect to Ollama at {self.base_url!r}."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                f"Ollama generation request failed: {exc}"
            ) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response_payload = self._parse_json(response)

        return self._build_generation_result(
            request=request,
            response_payload=response_payload,
            latency_ms=latency_ms,
        )

    def _extract_prompt(self, request: GenerationRequest) -> str:
        """Extract the prepared prompt from request metadata.

        ``GenerationRequest.context`` intentionally remains a structured
        list of RetrievedChunk objects. Prompt construction is performed
        separately by PromptBuilder. The resulting prompt is carried in
        request metadata under ``prompt``.
        """
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
        """Build the Ollama generation request payload."""
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        if self.max_tokens is not None:
            payload["options"]["num_predict"] = self.max_tokens

        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive

        return payload

    @staticmethod
    def _parse_json(response: requests.Response) -> dict[str, Any]:
        """Parse and validate an Ollama JSON response."""
        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama returned a response that is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise OllamaResponseError(
                "Ollama response must be a JSON object."
            )

        error_message = payload.get("error")

        if error_message:
            raise OllamaResponseError(
                f"Ollama returned an error: {error_message}"
            )

        return payload

    @staticmethod
    def _build_generation_result(
        *,
        request: GenerationRequest,
        response_payload: dict[str, Any],
        latency_ms: float,
    ) -> GenerationResult:
        """Convert an Ollama response into the project result model."""
        generated_code = response_payload.get("response")

        if not isinstance(generated_code, str):
            raise OllamaResponseError(
                "Ollama response does not contain a valid 'response' field."
            )

        generated_code = generated_code.strip()

        if not generated_code:
            raise OllamaResponseError(
                "Ollama returned an empty generation response."
            )

        prompt_tokens = LocalOllamaProvider._extract_optional_int(
            response_payload,
            "prompt_eval_count",
        )

        completion_tokens = LocalOllamaProvider._extract_optional_int(
            response_payload,
            "eval_count",
        )

        retrieved_chunk_ids = [
            chunk.chunk.chunk_id
            for chunk in request.context
        ]

        metadata = {
            "done": response_payload.get("done"),
            "done_reason": response_payload.get("done_reason"),
            "total_duration_ns": response_payload.get("total_duration"),
            "load_duration_ns": response_payload.get("load_duration"),
            "prompt_eval_duration_ns": response_payload.get(
                "prompt_eval_duration"
            ),
            "eval_duration_ns": response_payload.get("eval_duration"),
        }

        return GenerationResult(
            request_id=request.request_id,
            generated_code=generated_code,
            model_name=request.model_name or "unknown",
            provider_name=request.provider_name or "ollama",
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
        """Return an integer response field when it is available."""
        value = payload.get(key)

        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value

        return None

    def get_metadata(self) -> dict[str, Any]:
        """Return provider metadata without exposing secrets."""
        return {
            **super().get_metadata(),
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "keep_alive": self.keep_alive,
        }