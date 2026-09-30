"""Unit tests for the SecureCodeRAG generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from src.config import load_config
from src.generation.pipeline import (
    GenerationPipeline,
    GenerationPipelineExecutionError,
    GenerationPipelineInputError,
    PipelineRequest,
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
from src.retrieval.context_builder import BuiltContext
from src.retrieval.retriever import RetrievalResponse


def make_chunk(
    chunk_id: str = "chunk-1",
    content: str = "def hello():\n    return 'hello'",
) -> CodeChunk:
    """Create a deterministic code chunk for testing."""

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
        trust=SourceTrust.TRUSTED,
    )


def make_retrieved_chunk(
    chunk_id: str = "chunk-1",
) -> RetrievedChunk:
    """Create a deterministic retrieved chunk."""

    return RetrievedChunk(
        chunk=make_chunk(chunk_id=chunk_id),
        retrieval_score=0.95,
        rank=1,
        retriever_name="fake-retriever",
    )


def make_retrieval_response() -> RetrievalResponse:
    """Create a deterministic retrieval response."""

    return RetrievalResponse(
        query="Create a hello function.",
        results=(make_retrieved_chunk(),),
        requested_top_k=5,
        retriever_name="fake-retriever",
        embedding_model="fake-embedding-model",
        metadata={"test": True},
    )


class FakeRetriever:
    """Minimal retriever-compatible fake for pipeline tests."""

    def __init__(
        self,
        response: RetrievalResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or make_retrieval_response()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Return the configured response."""

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "metadata_filter": metadata_filter or {},
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class FakeContextBuilder:
    """Minimal context-builder fake."""

    def __init__(
        self,
        error: Exception | None = None,
    ) -> None:
        self.error = error
        self.calls: list[RetrievalResponse] = []

    def build_from_response(
        self,
        response: RetrievalResponse,
    ) -> BuiltContext:
        """Build deterministic context."""

        self.calls.append(response)

        if self.error is not None:
            raise self.error

        return BuiltContext(
            text="def hello():\n    return 'hello'",
            sources=(
                make_retrieved_chunk(),
            ),
            query=response.query,
            total_characters=32,
            total_lines=2,
            truncated=False,
            metadata={"test": True},
        )


class FakePromptBuilder:
    """Minimal prompt-builder fake."""

    def __init__(
        self,
        error: Exception | None = None,
    ) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def build_from_context(
        self,
        request: GenerationRequest,
        context: str,
    ) -> str:
        """Build a deterministic prompt."""

        self.calls.append(
            {
                "request": request,
                "context": context,
            }
        )

        if self.error is not None:
            raise self.error

        return (
            "SYSTEM: Use repository context safely.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "TASK:\n"
            f"{request.query}"
        )


@dataclass
class FakeProvider:
    """Minimal generation-provider fake."""

    model_name: str = "fake-model"
    provider_name: str = "fake-provider"

    def __post_init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Return deterministic generated code."""

        self.requests.append(request)

        return GenerationResult(
            request_id=request.request_id,
            generated_code="def hello():\n    return 'hello'",
            model_name=self.model_name,
            provider_name=self.provider_name,
            status=GenerationStatus.SUCCESS,
            prompt_tokens=20,
            completion_tokens=10,
            latency_ms=2.5,
            retrieved_chunk_ids=[
                item.chunk.chunk_id
                for item in request.context
            ],
            metadata={
                "fake": True,
            },
        )


@pytest.fixture
def config():
    """Load the project's default configuration."""

    return load_config()


@pytest.fixture
def pipeline_dependencies():
    """Create deterministic pipeline dependencies."""

    retriever = FakeRetriever()
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()
    provider = FakeProvider()

    return (
        retriever,
        context_builder,
        prompt_builder,
        provider,
    )


def make_pipeline_request() -> PipelineRequest:
    """Create a valid pipeline request."""

    return PipelineRequest(
        request_id="pipeline-test-001",
        task="code_generation",
        query="Create a hello function.",
        language=ProgrammingLanguage.PYTHON,
        top_k=5,
        metadata_filter={"repository_id": "repo-1"},
        defense_enabled=True,
        metadata={
            "experiment_condition": "A",
        },
    )


def test_pipeline_request_accepts_valid_input() -> None:
    """Valid pipeline requests should be accepted."""

    request = make_pipeline_request()

    assert request.request_id == "pipeline-test-001"
    assert request.task == "code_generation"
    assert request.language is ProgrammingLanguage.PYTHON
    assert request.top_k == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", ""),
        ("task", ""),
        ("query", ""),
    ],
)
def test_pipeline_request_rejects_empty_required_fields(
    field: str,
    value: str,
) -> None:
    """Required request fields must not be empty."""

    values = {
        "request_id": "pipeline-test-001",
        "task": "code_generation",
        "query": "Create a hello function.",
    }
    values[field] = value

    with pytest.raises(GenerationPipelineInputError):
        PipelineRequest(**values)


def test_pipeline_request_rejects_invalid_top_k() -> None:
    """top_k must be positive."""

    with pytest.raises(GenerationPipelineInputError):
        PipelineRequest(
            request_id="pipeline-test-001",
            task="code_generation",
            query="Create a hello function.",
            top_k=0,
        )


def test_pipeline_rejects_invalid_retriever(config) -> None:
    """Pipeline construction requires a retriever with retrieve()."""

    with pytest.raises(GenerationPipelineInputError):
        GenerationPipeline(
            config=config,
            retriever=object(),
        )


def test_pipeline_generate_orchestrates_all_stages(
    config,
    pipeline_dependencies,
    monkeypatch,
) -> None:
    """The pipeline should execute retrieval, context, prompt, and generation."""

    (
        retriever,
        context_builder,
        prompt_builder,
        provider,
    ) = pipeline_dependencies

    def fake_create(
        _config,
        provider_name=None,
        **_overrides,
    ):
        assert provider_name is None
        return provider

    monkeypatch.setattr(
        "src.generation.pipeline.GenerationFactory.create",
        fake_create,
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    request = make_pipeline_request()
    result = pipeline.generate(request)

    assert result.request == request
    assert result.generation.generated_code == (
        "def hello():\n    return 'hello'"
    )
    assert result.generation.status is GenerationStatus.SUCCESS
    assert result.prompt.startswith("SYSTEM:")
    assert result.latency_ms >= 0.0

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["query"] == request.query
    assert retriever.calls[0]["top_k"] == 5
    assert retriever.calls[0]["metadata_filter"] == {
        "repository_id": "repo-1",
    }

    assert len(context_builder.calls) == 1
    assert len(prompt_builder.calls) == 1
    assert len(provider.requests) == 1


def test_pipeline_preserves_retrieved_chunk_ids(
    config,
    pipeline_dependencies,
    monkeypatch,
) -> None:
    """Retrieved chunk IDs should survive through generation."""

    retriever, context_builder, prompt_builder, provider = (
        pipeline_dependencies
    )

    monkeypatch.setattr(
        "src.generation.pipeline.GenerationFactory.create",
        lambda *_args, **_kwargs: provider,
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    result = pipeline.generate(make_pipeline_request())

    assert result.retrieved_chunk_ids == ["chunk-1"]
    assert result.generation.retrieved_chunk_ids == ["chunk-1"]

    generation_request = provider.requests[0]

    assert generation_request.context[0].chunk.chunk_id == "chunk-1"
    assert generation_request.metadata["retrieved_chunk_ids"] == [
        "chunk-1"
    ]


def test_pipeline_includes_prompt_in_generation_metadata(
    config,
    pipeline_dependencies,
    monkeypatch,
) -> None:
    """The final provider request must contain the constructed prompt."""

    retriever, context_builder, prompt_builder, provider = (
        pipeline_dependencies
    )

    monkeypatch.setattr(
        "src.generation.pipeline.GenerationFactory.create",
        lambda *_args, **_kwargs: provider,
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    result = pipeline.generate(make_pipeline_request())

    generation_request = provider.requests[0]

    assert generation_request.metadata["prompt"] == result.prompt
    assert "Create a hello function." in result.prompt
    assert "def hello()" in result.prompt


def test_pipeline_preserves_defense_setting(
    config,
    pipeline_dependencies,
    monkeypatch,
) -> None:
    """Defense state must be propagated into GenerationRequest."""

    retriever, context_builder, prompt_builder, provider = (
        pipeline_dependencies
    )

    monkeypatch.setattr(
        "src.generation.pipeline.GenerationFactory.create",
        lambda *_args, **_kwargs: provider,
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    request = PipelineRequest(
        request_id="defense-off",
        task="code_generation",
        query="Create a function.",
        defense_enabled=False,
    )

    pipeline.generate(request)

    generation_request = provider.requests[0]

    assert generation_request.defense_enabled is False
    assert generation_request.metadata["defense_enabled"] is False


def test_retrieve_only_skips_generation(
    config,
) -> None:
    """retrieve_only should not invoke an LLM."""

    retriever = FakeRetriever()

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
    )

    result = pipeline.retrieve_only(make_pipeline_request())

    assert isinstance(result, RetrievalResponse)
    assert len(result.results) == 1
    assert retriever.calls


def test_build_context_only_skips_generation(
    config,
) -> None:
    """build_context_only should stop after context construction."""

    retriever = FakeRetriever()
    context_builder = FakeContextBuilder()

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
    )

    result = pipeline.build_context_only(make_pipeline_request())

    assert isinstance(result, BuiltContext)
    assert result.text == "def hello():\n    return 'hello'"
    assert result.query == "Create a hello function."
    assert len(context_builder.calls) == 1


def test_build_prompt_only_skips_generation(
    config,
) -> None:
    """build_prompt_only should stop before provider execution."""

    retriever = FakeRetriever()
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    result = pipeline.build_prompt_only(make_pipeline_request())

    assert result.startswith("SYSTEM:")
    assert "def hello()" in result
    assert len(prompt_builder.calls) == 1


def test_pipeline_wraps_retrieval_failure(
    config,
) -> None:
    """Retriever failures should be wrapped by the pipeline."""

    retriever = FakeRetriever(
        error=RuntimeError("vector store unavailable"),
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
    )

    with pytest.raises(GenerationPipelineExecutionError) as exc_info:
        pipeline.generate(make_pipeline_request())

    assert "Retrieval stage failed." in str(exc_info.value)


def test_pipeline_wraps_context_failure(
    config,
) -> None:
    """Context-builder failures should be wrapped."""

    retriever = FakeRetriever()
    context_builder = FakeContextBuilder(
        error=RuntimeError("context failure"),
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
    )

    with pytest.raises(GenerationPipelineExecutionError) as exc_info:
        pipeline.generate(make_pipeline_request())

    assert "Context-building stage failed." in str(exc_info.value)


def test_pipeline_wraps_prompt_failure(
    config,
) -> None:
    """Prompt-builder failures should be wrapped."""

    retriever = FakeRetriever()
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder(
        error=RuntimeError("prompt failure"),
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    with pytest.raises(GenerationPipelineExecutionError) as exc_info:
        pipeline.generate(make_pipeline_request())

    assert "Prompt-building stage failed." in str(exc_info.value)


def test_pipeline_wraps_generation_failure(
    config,
    monkeypatch,
) -> None:
    """Provider failures should be wrapped by the pipeline."""

    retriever = FakeRetriever()
    context_builder = FakeContextBuilder()
    prompt_builder = FakePromptBuilder()

    class FailingProvider:
        """Provider that always fails."""

        def generate(
            self,
            _request: GenerationRequest,
        ) -> GenerationResult:
            raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        "src.generation.pipeline.GenerationFactory.create",
        lambda *_args, **_kwargs: FailingProvider(),
    )

    pipeline = GenerationPipeline(
        config=config,
        retriever=retriever,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
    )

    with pytest.raises(GenerationPipelineExecutionError) as exc_info:
        pipeline.generate(make_pipeline_request())

    assert "Generation pipeline execution failed." in str(
        exc_info.value
    )