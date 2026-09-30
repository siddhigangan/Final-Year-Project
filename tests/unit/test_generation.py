"""Unit tests for the SecureCodeRAG generation layer."""

from __future__ import annotations

from typing import Any

import pytest

from src.config import load_config
from src.generation.base import (
    GenerationConfigurationError,
    GenerationInputError,
    GenerationProvider,
)
from src.generation.factory import GenerationFactory
from src.generation.hosted_openai import (
    HostedOpenAIConnectionError,
    HostedOpenAIProvider,
    HostedOpenAIResponseError,
)
from src.generation.local_ollama import (
    LocalOllamaProvider,
    OllamaConnectionError,
    OllamaResponseError,
)
from src.models import (
    CodeChunk,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    ProgrammingLanguage,
    RetrievedChunk,
    SourceTrust,
)


class FakeResponse:
    """Minimal requests.Response replacement for unit tests."""

    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self) -> None:
        """Raise an HTTP error for unsuccessful responses."""
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        """Return the configured JSON payload."""
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class DummyProvider(GenerationProvider):
    """Concrete provider used to test the abstract base class."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a deterministic test result."""
        request = self.prepare_request(request)

        return GenerationResult(
            request_id=request.request_id,
            generated_code="print('hello')",
            model_name=self.model_name,
            provider_name=self.provider_name,
            status=GenerationStatus.SUCCESS,
        )


def make_chunk(
    chunk_id: str = "chunk-1",
    *,
    content: str = "def hello():\n    return 'hello'",
) -> CodeChunk:
    """Create a valid code chunk for generation tests."""
    return CodeChunk(
        chunk_id=chunk_id,
        source_file_id="file-1",
        repository_id="repo-1",
        content=content,
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=2,
        symbol_name="hello",
        symbol_type="function",
        ast_node_type="function_definition",
        parent_symbol=None,
        is_documentation=False,
        trust=SourceTrust.TRUSTED,
        metadata={"relative_path": "src/example.py"},
    )


def make_retrieved_chunk(
    chunk_id: str = "chunk-1",
) -> RetrievedChunk:
    """Create a valid retrieved chunk."""
    return RetrievedChunk(
        chunk=make_chunk(chunk_id),
        retrieval_score=0.95,
        rank=1,
        retriever_name="test-retriever",
    )


def make_request(
    *,
    request_id: str = "request-1",
    model_name: str = "qwen2.5-coder:7b",
    provider_name: str = "ollama",
    prompt: str = "Generate a Python hello-world function.",
) -> GenerationRequest:
    """Create a valid generation request."""
    retrieved_chunk = make_retrieved_chunk()

    return GenerationRequest(
        request_id=request_id,
        task="code_generation",
        query="Generate a Python hello-world function.",
        context=[retrieved_chunk],
        language=ProgrammingLanguage.PYTHON,
        model_name=model_name,
        provider_name=provider_name,
        defense_enabled=True,
        metadata={
            "prompt": prompt,
        },
    )


def test_generation_provider_rejects_empty_model() -> None:
    """Providers must reject empty model names."""
    with pytest.raises(GenerationConfigurationError):
        DummyProvider(model_name="")


def test_generation_provider_uses_default_provider_name() -> None:
    """Providers should use their class-level provider name by default."""
    provider = DummyProvider(model_name="test-model")

    assert provider.provider_name == "unknown"


def test_generation_provider_validates_request_type() -> None:
    """Providers must reject objects that are not GenerationRequest."""
    provider = DummyProvider(model_name="test-model")

    with pytest.raises(GenerationInputError):
        provider.generate("invalid request")  # type: ignore[arg-type]


def test_dummy_provider_returns_generation_result() -> None:
    """The base contract should support GenerationResult outputs."""
    provider = DummyProvider(
        model_name="test-model",
        provider_name="test-provider",
    )

    result = provider.generate(make_request())

    assert isinstance(result, GenerationResult)
    assert result.request_id == "request-1"
    assert result.generated_code == "print('hello')"
    assert result.model_name == "test-model"
    assert result.provider_name == "test-provider"
    assert result.status == GenerationStatus.SUCCESS


def test_local_ollama_provider_configuration() -> None:
    """Ollama provider should expose its configured metadata."""
    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    metadata = provider.get_metadata()

    assert metadata["provider_name"] == "ollama"
    assert metadata["model_name"] == "qwen2.5-coder:7b"
    assert metadata["base_url"] == "http://localhost:11434"
    assert metadata["supports_streaming"] is True


def test_local_ollama_provider_rejects_invalid_timeout() -> None:
    """Ollama provider should reject non-positive timeouts."""
    with pytest.raises(GenerationConfigurationError):
        LocalOllamaProvider(
            model_name="qwen2.5-coder:7b",
            timeout_seconds=0,
        )


def test_local_ollama_provider_rejects_invalid_temperature() -> None:
    """Ollama provider should reject negative temperatures."""
    with pytest.raises(GenerationConfigurationError):
        LocalOllamaProvider(
            model_name="qwen2.5-coder:7b",
            temperature=-0.1,
        )


def test_local_ollama_provider_requires_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama generation should require a prepared prompt."""
    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    request = make_request()
    request = GenerationRequest(
        request_id=request.request_id,
        task=request.task,
        query=request.query,
        context=request.context,
        language=request.language,
        model_name=request.model_name,
        provider_name=request.provider_name,
        defense_enabled=request.defense_enabled,
        metadata={},
    )

    monkeypatch.setattr(
        "src.generation.local_ollama.requests.post",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(GenerationInputError):
        provider.generate(request)


def test_local_ollama_generate_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama responses should map to GenerationResult."""
    payload = {
        "model": "qwen2.5-coder:7b",
        "response": "def hello():\n    return 'hello'",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 20,
        "eval_count": 12,
        "total_duration": 1_000_000,
    }

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(payload)

    monkeypatch.setattr(
        "src.generation.local_ollama.requests.post",
        fake_post,
    )

    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    result = provider.generate(make_request())

    assert isinstance(result, GenerationResult)
    assert result.generated_code.startswith("def hello")
    assert result.model_name == "qwen2.5-coder:7b"
    assert result.provider_name == "ollama"
    assert result.status == GenerationStatus.SUCCESS
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 12
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    assert result.retrieved_chunk_ids == ["chunk-1"]
    assert result.metadata["done"] is True


def test_local_ollama_generate_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama invalid JSON should produce a response error."""

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(
            None,
            json_error=ValueError("invalid json"),
        )

    monkeypatch.setattr(
        "src.generation.local_ollama.requests.post",
        fake_post,
    )

    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    with pytest.raises(OllamaResponseError):
        provider.generate(make_request())


def test_local_ollama_generate_rejects_missing_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama responses without generated text should fail."""

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse({"done": True})

    monkeypatch.setattr(
        "src.generation.local_ollama.requests.post",
        fake_post,
    )

    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    with pytest.raises(OllamaResponseError):
        provider.generate(make_request())


def test_local_ollama_generate_handles_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama connection failures should be wrapped."""

    def fake_post(*args: Any, **kwargs: Any) -> None:
        import requests

        raise requests.ConnectionError("server unavailable")

    monkeypatch.setattr(
        "src.generation.local_ollama.requests.post",
        fake_post,
    )

    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    with pytest.raises(OllamaConnectionError):
        provider.generate(make_request())


def test_local_ollama_list_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama model listing should extract model names."""

    payload = {
        "models": [
            {"name": "qwen2.5-coder:7b"},
            {"name": "qwen2.5-coder:latest"},
        ]
    }

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(payload)

    monkeypatch.setattr(
        "src.generation.local_ollama.requests.get",
        fake_get,
    )

    provider = LocalOllamaProvider(
        model_name="qwen2.5-coder:7b",
    )

    assert provider.list_models() == [
        "qwen2.5-coder:7b",
        "qwen2.5-coder:latest",
    ]


def test_hosted_openai_provider_configuration() -> None:
    """OpenAI provider should expose safe metadata."""
    provider = HostedOpenAIProvider(
        model_name="gpt-4o-mini",
        api_key="secret-test-key",
    )

    metadata = provider.get_metadata()

    assert metadata["provider_name"] == "openai"
    assert metadata["model_name"] == "gpt-4o-mini"
    assert metadata["base_url"] == "https://api.openai.com/v1"
    assert metadata["api_key_configured"] is True
    assert "secret-test-key" not in str(metadata)


def test_hosted_openai_provider_rejects_empty_api_key() -> None:
    """OpenAI provider must require an API key."""
    with pytest.raises(GenerationConfigurationError):
        HostedOpenAIProvider(
            model_name="gpt-4o-mini",
            api_key="",
        )


def test_hosted_openai_generate_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI-compatible responses should map to GenerationResult."""
    payload = {
        "id": "chatcmpl-test",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "def hello():\n    return 'hello'",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
        },
    }

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        assert kwargs["headers"]["Authorization"] == "Bearer secret-key"
        return FakeResponse(payload)

    monkeypatch.setattr(
        "src.generation.hosted_openai.requests.post",
        fake_post,
    )

    provider = HostedOpenAIProvider(
        model_name="gpt-4o-mini",
        api_key="secret-key",
    )

    result = provider.generate(
        make_request(
            model_name="gpt-4o-mini",
            provider_name="openai",
        )
    )

    assert isinstance(result, GenerationResult)
    assert result.generated_code.startswith("def hello")
    assert result.model_name == "gpt-4o-mini"
    assert result.provider_name == "openai"
    assert result.status == GenerationStatus.SUCCESS
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 15
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    assert result.retrieved_chunk_ids == ["chunk-1"]
    assert result.metadata["finish_reason"] == "stop"


def test_hosted_openai_generate_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI responses without choices should fail."""

    def fake_post(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse({"id": "test"})

    monkeypatch.setattr(
        "src.generation.hosted_openai.requests.post",
        fake_post,
    )

    provider = HostedOpenAIProvider(
        model_name="gpt-4o-mini",
        api_key="secret-key",
    )

    with pytest.raises(HostedOpenAIResponseError):
        provider.generate(
            make_request(
                model_name="gpt-4o-mini",
                provider_name="openai",
            )
        )


def test_hosted_openai_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI connection failures should be wrapped."""

    def fake_post(*args: Any, **kwargs: Any) -> None:
        import requests

        raise requests.ConnectionError("network unavailable")

    monkeypatch.setattr(
        "src.generation.hosted_openai.requests.post",
        fake_post,
    )

    provider = HostedOpenAIProvider(
        model_name="gpt-4o-mini",
        api_key="secret-key",
    )

    with pytest.raises(HostedOpenAIConnectionError):
        provider.generate(
            make_request(
                model_name="gpt-4o-mini",
                provider_name="openai",
            )
        )


def test_generation_factory_supports_expected_providers() -> None:
    """Factory should expose the configured provider types."""
    assert GenerationFactory.supported_providers() == (
        "ollama",
        "openai",
    )


def test_generation_factory_creates_ollama_provider() -> None:
    """Factory should create the local Ollama provider."""
    config = load_config()

    provider = GenerationFactory.create(
        config,
        provider_name="ollama",
        model_name="qwen2.5-coder:7b",
    )

    assert isinstance(provider, LocalOllamaProvider)
    assert provider.model_name == "qwen2.5-coder:7b"


def test_generation_factory_rejects_unknown_provider() -> None:
    """Factory should reject unsupported provider names."""
    config = load_config()

    with pytest.raises(GenerationConfigurationError):
        GenerationFactory.create(
            config,
            provider_name="unsupported-provider",
        )


def test_generation_factory_creates_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should create OpenAI from environment configuration."""
    monkeypatch.setenv("OPENAI_API_KEY", "factory-test-key")

    config = load_config()

    provider = GenerationFactory.create(
        config,
        provider_name="openai",
        model_name="gpt-4o-mini",
    )

    assert isinstance(provider, HostedOpenAIProvider)
    assert provider.model_name == "gpt-4o-mini"

    metadata = provider.get_metadata()

    assert metadata["api_key_configured"] is True
    assert "factory-test-key" not in str(metadata)


def test_generation_factory_requires_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should reject OpenAI configuration without an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = load_config()

    with pytest.raises(GenerationConfigurationError):
        GenerationFactory.create(
            config,
            provider_name="openai",
            model_name="gpt-4o-mini",
        )