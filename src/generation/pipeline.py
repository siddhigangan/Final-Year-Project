"""End-to-end retrieval-augmented code generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from src.config import AppConfig
from src.models import (
    GenerationRequest,
    GenerationResult,
    ProgrammingLanguage,
)
from src.retrieval.context_builder import BuiltContext, ContextBuilder
from src.retrieval.retriever import RetrievalResponse

from .factory import GenerationFactory
from .prompt_builder import PromptBuilder


class GenerationPipelineError(RuntimeError):
    """Base exception for generation-pipeline failures."""


class GenerationPipelineInputError(GenerationPipelineError):
    """Raised when pipeline input is invalid."""


class GenerationPipelineExecutionError(GenerationPipelineError):
    """Raised when a pipeline stage fails during execution."""


@dataclass(frozen=True)
class PipelineRequest:
    """Input required to execute the generation pipeline."""

    request_id: str
    task: str
    query: str
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    top_k: int | None = None
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    defense_enabled: bool = True
    provider_name: str | None = None
    model_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise GenerationPipelineInputError(
                "request_id cannot be empty."
            )

        if not self.task.strip():
            raise GenerationPipelineInputError(
                "task cannot be empty."
            )

        if not self.query.strip():
            raise GenerationPipelineInputError(
                "query cannot be empty."
            )

        if self.top_k is not None and self.top_k <= 0:
            raise GenerationPipelineInputError(
                "top_k must be greater than zero when provided."
            )


@dataclass(frozen=True)
class PipelineResult:
    """Complete output of one RAG generation execution."""

    request: PipelineRequest
    retrieval: RetrievalResponse
    context: BuiltContext
    generation: GenerationResult
    prompt: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        """Return IDs of chunks retrieved for this request."""

        return [
            item.chunk.chunk_id
            for item in self.retrieval.results
        ]


class GenerationPipeline:
    """Coordinate retrieval, context construction, prompting, and generation."""

    def __init__(
        self,
        config: AppConfig,
        retriever: Any,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        if not isinstance(config, AppConfig):
            raise GenerationPipelineInputError(
                "config must be an AppConfig instance."
            )

        if not callable(getattr(retriever, "retrieve", None)):
            raise GenerationPipelineInputError(
                "retriever must provide a callable retrieve method."
            )

        self.config = config
        self.retriever = retriever
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(
        self,
        request: PipelineRequest,
    ) -> PipelineResult:
        """Execute the complete retrieval-augmented generation workflow."""

        if not isinstance(request, PipelineRequest):
            raise GenerationPipelineInputError(
                "request must be a PipelineRequest instance."
            )

        started = perf_counter()

        try:
            retrieval = self._retrieve(request)

            context = self._build_context(retrieval)

            prompt = self._build_prompt(
                request=request,
                context=context,
            )

            generation_request = self._build_generation_request(
                request=request,
                retrieval=retrieval,
                context=context,
                prompt=prompt,
            )

            factory_kwargs: dict[str, Any] = {}

            if request.provider_name is not None:
                factory_kwargs["provider_name"] = request.provider_name

            if request.model_name is not None:
                factory_kwargs["model_name"] = request.model_name

            provider = GenerationFactory.create(
                self.config,
                **factory_kwargs,
            )

            generation = provider.generate(generation_request)

        except GenerationPipelineError:
            raise
        except Exception as exc:
            raise GenerationPipelineExecutionError(
                "Generation pipeline execution failed."
            ) from exc

        latency_ms = (perf_counter() - started) * 1000.0

        pipeline_metadata = {
            "pipeline_version": "0.1.0",
            "retrieved_chunk_count": len(retrieval.results),
            "context_source_count": len(context.sources),
            "provider_name": generation.provider_name,
            "model_name": generation.model_name,
            "defense_enabled": request.defense_enabled,
            **request.metadata,
        }

        return PipelineResult(
            request=request,
            retrieval=retrieval,
            context=context,
            generation=generation,
            prompt=prompt,
            latency_ms=latency_ms,
            metadata=pipeline_metadata,
        )

    def _retrieve(
        self,
        request: PipelineRequest,
    ) -> RetrievalResponse:
        """Retrieve relevant repository context."""

        try:
            return self.retriever.retrieve(
                query=request.query,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
        except TypeError as exc:
            raise GenerationPipelineExecutionError(
                "Retriever interface is incompatible with the generation "
                "pipeline."
            ) from exc
        except Exception as exc:
            raise GenerationPipelineExecutionError(
                "Retrieval stage failed."
            ) from exc

    def _build_context(
        self,
        retrieval: RetrievalResponse,
    ) -> BuiltContext:
        """Convert retrieved chunks into structured generation context."""

        try:
            return self.context_builder.build_from_response(retrieval)
        except Exception as exc:
            raise GenerationPipelineExecutionError(
                "Context-building stage failed."
            ) from exc

    def _build_prompt(
        self,
        request: PipelineRequest,
        context: BuiltContext,
    ) -> str:
        """Build the final security-aware generation prompt."""

        try:
            return self.prompt_builder.build_from_context(
                request=self._create_prompt_request(request),
                context=context.text,
            )
        except Exception as exc:
            raise GenerationPipelineExecutionError(
                "Prompt-building stage failed."
            ) from exc

    def _create_prompt_request(
        self,
        request: PipelineRequest,
    ) -> GenerationRequest:
        """Create a temporary request for prompt construction."""

        return GenerationRequest(
            request_id=request.request_id,
            task=request.task,
            query=request.query,
            context=[],
            language=request.language,
            model_name=request.model_name or self.config.code_llm,
            provider_name=(
                request.provider_name
                or self.config.code_llm_provider
            ),
            defense_enabled=request.defense_enabled,
            metadata=dict(request.metadata),
        )

    def _build_generation_request(
        self,
        request: PipelineRequest,
        retrieval: RetrievalResponse,
        context: BuiltContext,
        prompt: str,
    ) -> GenerationRequest:
        """Build the provider-independent generation request."""

        generation_metadata = {
            **request.metadata,
            "prompt": prompt,
            "pipeline_version": "0.1.0",
            "retriever_name": retrieval.retriever_name,
            "retrieved_chunk_ids": [
                item.chunk.chunk_id
                for item in retrieval.results
            ],
            "retrieved_chunk_count": len(retrieval.results),
            "context_source_count": len(context.sources),
            "defense_enabled": request.defense_enabled,
        }

        return GenerationRequest(
            request_id=request.request_id,
            task=request.task,
            query=request.query,
            context=list(retrieval.results),
            language=request.language,
            model_name=request.model_name or self.config.code_llm,
            provider_name=(
                request.provider_name
                or self.config.code_llm_provider
            ),
            defense_enabled=request.defense_enabled,
            metadata=generation_metadata,
        )

    def retrieve_only(
        self,
        request: PipelineRequest,
    ) -> RetrievalResponse:
        """Run only the retrieval stage."""

        if not isinstance(request, PipelineRequest):
            raise GenerationPipelineInputError(
                "request must be a PipelineRequest instance."
            )

        return self._retrieve(request)

    def build_context_only(
        self,
        request: PipelineRequest,
    ) -> BuiltContext:
        """Run retrieval and context construction without generation."""

        if not isinstance(request, PipelineRequest):
            raise GenerationPipelineInputError(
                "request must be a PipelineRequest instance."
            )

        retrieval = self._retrieve(request)

        return self._build_context(retrieval)

    def build_prompt_only(
        self,
        request: PipelineRequest,
    ) -> str:
        """Run retrieval, context construction, and prompt construction."""

        if not isinstance(request, PipelineRequest):
            raise GenerationPipelineInputError(
                "request must be a PipelineRequest instance."
            )

        retrieval = self._retrieve(request)
        context = self._build_context(retrieval)

        return self._build_prompt(
            request=request,
            context=context,
        )