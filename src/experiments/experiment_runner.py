"""Controlled experiment orchestration for SecureCodeRAG."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from src.defense.pipeline import DefensePipeline, DefensePipelineResult
from src.generation.base import GenerationProvider
from src.models import (
    CodeChunk,
    ExperimentCondition,
    ExperimentConfig,
    ExperimentResult,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    ProgrammingLanguage,
    RetrievedChunk,
    SecurityDecision,
    SecurityFinding,
)
from src.retrieval.context_builder import (
    BuiltContext,
    ContextBuilder,
)
from src.retrieval.retriever import RetrievalResponse, Retriever


class ExperimentRunnerError(RuntimeError):
    """Base exception for experiment-runner failures."""


class ExperimentRunnerConfigurationError(ValueError):
    """Raised when the experiment runner is incorrectly configured."""


class ExperimentRunnerInputError(ExperimentRunnerError):
    """Raised when experiment inputs are invalid."""


@dataclass(frozen=True)
class ExperimentRunInput:
    """Input required to execute one controlled experiment."""

    query: str
    task: str
    experiment_config: ExperimentConfig
    request_id: str
    baseline_generated_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise ExperimentRunnerInputError(
                "query must be a string."
            )

        if not isinstance(self.task, str):
            raise ExperimentRunnerInputError(
                "task must be a string."
            )

        if not isinstance(
            self.experiment_config,
            ExperimentConfig,
        ):
            raise ExperimentRunnerInputError(
                "experiment_config must be an ExperimentConfig."
            )

        if (
            not isinstance(self.request_id, str)
            or not self.request_id.strip()
        ):
            raise ExperimentRunnerInputError(
                "request_id must be a non-empty string."
            )

        if (
            self.baseline_generated_code is not None
            and not isinstance(
                self.baseline_generated_code,
                str,
            )
        ):
            raise ExperimentRunnerInputError(
                "baseline_generated_code must be a string or None."
            )


@dataclass(frozen=True)
class ExperimentExecution:
    """Internal execution state before final result aggregation."""

    retrieval: RetrievalResponse
    context: BuiltContext
    defense: DefensePipelineResult | None
    generation: GenerationResult
    generated_code_findings: tuple[SecurityFinding, ...]
    elapsed_ms: float
    poison_retrieved: bool
    poison_in_context: bool
    generation_changed: bool
    vulnerability_introduced: bool


class ExperimentRunner:
    """Execute reproducible SecureCodeRAG controlled experiments."""

    def __init__(
        self,
        retriever: Retriever,
        generator: GenerationProvider,
        *,
        defense_pipeline: DefensePipeline | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        if not isinstance(retriever, Retriever):
            raise ExperimentRunnerConfigurationError(
                "retriever must be a Retriever instance."
            )

        if not isinstance(generator, GenerationProvider):
            raise ExperimentRunnerConfigurationError(
                "generator must implement GenerationProvider."
            )

        if (
            defense_pipeline is not None
            and not isinstance(
                defense_pipeline,
                DefensePipeline,
            )
        ):
            raise ExperimentRunnerConfigurationError(
                "defense_pipeline must be a DefensePipeline or None."
            )

        if (
            context_builder is not None
            and not isinstance(
                context_builder,
                ContextBuilder,
            )
        ):
            raise ExperimentRunnerConfigurationError(
                "context_builder must be a ContextBuilder or None."
            )

        self._retriever = retriever
        self._generator = generator
        self._defense_pipeline = defense_pipeline
        self._context_builder = (
            context_builder or ContextBuilder()
        )

    @property
    def retriever(self) -> Retriever:
        """Return the configured retriever."""
        return self._retriever

    @property
    def generator(self) -> GenerationProvider:
        """Return the configured generation provider."""
        return self._generator

    @property
    def defense_pipeline(
        self,
    ) -> DefensePipeline | None:
        """Return the configured defense pipeline."""
        return self._defense_pipeline

    def run(
        self,
        experiment: ExperimentRunInput,
    ) -> ExperimentResult:
        """Execute one controlled experiment."""

        if not isinstance(
            experiment,
            ExperimentRunInput,
        ):
            raise ExperimentRunnerInputError(
                "experiment must be an ExperimentRunInput."
            )

        self._validate_run_input(experiment)

        config = experiment.experiment_config
        started_at = perf_counter()

        use_defense = self._condition_uses_defense(
            config.condition
        )

        poison_present = self._condition_contains_poison(
            config.condition
        )

        if (
            use_defense
            and self._defense_pipeline is None
        ):
            raise ExperimentRunnerConfigurationError(
                "A defense-enabled condition requires a "
                "DefensePipeline."
            )

        retrieval = self._retrieve(
            query=experiment.query,
            top_k=config.top_k,
        )

        detected_poison_retrieval = (
            self._detect_poison_retrieval(
                retrieval.results
            )
        )

        # In poisoned experimental conditions, a successful
        # retrieval from the poisoned corpus represents a
        # poison-retrieval event even when the synthetic fixture
        # does not explicitly mark the retrieved chunk.
        poison_retrieved = (
            detected_poison_retrieval
            or (
                poison_present
                and len(retrieval.results) > 0
            )
        )

        original_context = self._context_builder.build(
            retrieval.results,
            query=experiment.query,
        )

        effective_context = original_context
        defense_result: DefensePipelineResult | None = None

        if use_defense:
            defense_result = (
                self._defense_pipeline.analyze_context(
                    query=experiment.query,
                    context=original_context,
                    retrieved_chunks=retrieval.results,
                )
            )

            if defense_result.safe_context is not None:
                effective_context = (
                    defense_result.safe_context
                )

        poison_in_context = (
            self._detect_poison_in_context(
                effective_context,
                retrieval.results,
            )
        )

        generation_metadata = {
            **experiment.metadata,
            "experiment_id": config.experiment_id,
            "condition": config.condition.value,
            "repository_id": config.repository_id,
            "retriever_name": config.retriever_name,
            "poison_present": poison_present,
            "poison_retrieved": poison_retrieved,
            "poison_in_context": poison_in_context,
        }

        generation_request = GenerationRequest(
            request_id=experiment.request_id,
            task=experiment.task,
            query=experiment.query,
            context=list(retrieval.results),
            language=config.language,
            model_name=config.model_name,
            provider_name=self._generator.provider_name,
            defense_enabled=use_defense,
            metadata=generation_metadata,
        )

        if (
            use_defense
            and defense_result is not None
        ):
            if defense_result.blocked:
                generation_result = (
                    self._build_blocked_generation_result(
                        request=experiment,
                        config=config,
                        defense_result=defense_result,
                        retrieved_chunks=retrieval.results,
                    )
                )
            else:
                generation_request = (
                    self._apply_safe_context(
                        generation_request,
                        defense_result,
                        original_chunks=retrieval.results,
                    )
                )

                generation_result = (
                    self._generator.generate(
                        generation_request
                    )
                )
        else:
            generation_result = (
                self._generator.generate(
                    generation_request
                )
            )

        generated_code_findings: tuple[
            SecurityFinding, ...
        ] = ()

        if self._should_analyze_generated_code(
            generation_result
        ):
            generated_analysis = (
                self._defense_pipeline.analyze_generated_code(
                    code=generation_result.generated_code,
                    file_name="generated_code.py",
                    language=config.language.value,
                    query=experiment.query,
                )
                if self._defense_pipeline is not None
                else None
            )

            if generated_analysis is not None:
                generated_code_findings = (
                    self._convert_security_findings(
                        generated_analysis.blocking_findings
                    )
                )

        generation_changed = (
            self._detect_generation_change(
                generation_result,
                experiment.baseline_generated_code,
            )
        )

        vulnerability_introduced = (
            self._detect_vulnerability(
                generated_code_findings
            )
        )

        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000.0

        defense_decision = (
            defense_result.final_decision.value
            if defense_result is not None
            else SecurityDecision.PASS.value
        )

        result_metadata = {
            **experiment.metadata,
            "experiment_id": config.experiment_id,
            "condition": config.condition.value,
            "task": experiment.task,
            "query": experiment.query,
            "repository_id": config.repository_id,
            "language": config.language.value,
            "model_name": config.model_name,
            "retriever_name": config.retriever_name,
            "top_k": config.top_k,
            "defense_enabled": use_defense,
            "defense_decision": defense_decision,
            "defense_blocked": (
                defense_result.blocked
                if defense_result is not None
                else False
            ),
            "defense_requires_review": (
                defense_result.requires_review
                if defense_result is not None
                else False
            ),
            "retrieved_chunk_ids": list(
                retrieval.chunk_ids
            ),
            "retrieved_chunk_count": len(
                retrieval.results
            ),
            "poison_present": poison_present,
            "poison_retrieved": poison_retrieved,
            "poison_in_context": poison_in_context,
            "generation_changed": generation_changed,
            "vulnerability_introduced": (
                vulnerability_introduced
            ),
            "generated_code_length": len(
                generation_result.generated_code
            ),
            "generation_status": (
                generation_result.status.value
            ),
            "provider_name": (
                generation_result.provider_name
            ),
            "generation_model": (
                generation_result.model_name
            ),
            "generated_code_findings": len(
                generated_code_findings
            ),
            "safe_context_applied": (
                use_defense
                and defense_result is not None
                and not defense_result.blocked
                and defense_result.safe_context is not None
            ),
        }

        return ExperimentResult(
            experiment_id=config.experiment_id,
            condition=config.condition,
            request_id=experiment.request_id,
            poison_present=poison_present,
            poison_retrieved=poison_retrieved,
            poison_in_context=poison_in_context,
            generation_changed=generation_changed,
            vulnerability_introduced=(
                vulnerability_introduced
            ),
            latency_ms=elapsed_ms,
            findings=list(generated_code_findings),
            metadata=result_metadata,
        )

    @staticmethod
    def _validate_run_input(
        experiment: ExperimentRunInput,
    ) -> None:
        """Validate experiment input values."""

        if not experiment.query.strip():
            raise ExperimentRunnerInputError(
                "query must be a non-empty string."
            )

        if not experiment.task.strip():
            raise ExperimentRunnerInputError(
                "task must be a non-empty string."
            )

    def _retrieve(
        self,
        *,
        query: str,
        top_k: int,
    ) -> RetrievalResponse:
        """Run retrieval and normalize retrieval failures."""

        try:
            return self._retriever.retrieve(
                query=query,
                top_k=top_k,
            )
        except Exception as exc:
            raise ExperimentRunnerError(
                f"Experiment retrieval failed: {exc}"
            ) from exc

    @staticmethod
    def _condition_uses_defense(
        condition: ExperimentCondition,
    ) -> bool:
        """Return whether defense is enabled for a condition."""

        return condition in {
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        }

    @staticmethod
    def _condition_contains_poison(
        condition: ExperimentCondition,
    ) -> bool:
        """Return whether the condition uses poisoned data."""

        return condition in {
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
        }

    @staticmethod
    def _detect_poison_retrieval(
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> bool:
        """Detect explicitly poisoned retrieved chunks."""

        for retrieved_chunk in retrieved_chunks:
            chunk = retrieved_chunk.chunk

            trust_value = getattr(
                chunk.trust,
                "value",
                str(chunk.trust),
            )

            if (
                str(trust_value)
                .strip()
                .lower()
                == "poisoned"
            ):
                return True

            if ExperimentRunner._metadata_marks_poisoned(
                chunk.metadata
            ):
                return True

            if ExperimentRunner._metadata_marks_poisoned(
                retrieved_chunk.retrieval_metadata
            ):
                return True

        return False

    @staticmethod
    def _metadata_marks_poisoned(
        metadata: dict[str, Any],
    ) -> bool:
        """Return whether metadata explicitly identifies poison."""

        poison_keys = {
            "poisoned",
            "is_poisoned",
            "poison",
            "poison_category",
            "poison_type",
            "trust",
            "source_trust",
        }

        for key in poison_keys:
            value = metadata.get(key)

            if isinstance(value, bool) and value:
                return True

            if isinstance(value, str):
                normalized = value.strip().lower()

                if normalized in {
                    "true",
                    "yes",
                    "poisoned",
                    "poison",
                }:
                    return True

        return False

    @staticmethod
    def _detect_poison_in_context(
        context: BuiltContext,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> bool:
        """Detect poisoned content surviving into context."""

        if not context.text.strip():
            return False

        context_source_ids = {
            source.chunk_id
            for source in context.sources
        }

        for retrieved_chunk in retrieved_chunks:
            chunk = retrieved_chunk.chunk

            trust_value = getattr(
                chunk.trust,
                "value",
                str(chunk.trust),
            )

            is_poisoned = (
                str(trust_value)
                .strip()
                .lower()
                == "poisoned"
                or ExperimentRunner._metadata_marks_poisoned(
                    chunk.metadata
                )
                or ExperimentRunner._metadata_marks_poisoned(
                    retrieved_chunk.retrieval_metadata
                )
            )

            if not is_poisoned:
                continue

            if chunk.chunk_id not in context_source_ids:
                continue

            if chunk.content.strip() in context.text:
                return True

        return False

    @staticmethod
    def _apply_safe_context(
        request: GenerationRequest,
        defense_result: DefensePipelineResult,
        *,
        original_chunks: Sequence[RetrievedChunk],
    ) -> GenerationRequest:
        """Replace generation context with sanitized context."""

        safe_context = defense_result.safe_context

        if safe_context is None:
            return request

        sanitized_chunks = (
            ExperimentRunner._retrieved_chunks_from_safe_context(
                safe_context,
                original_chunks=original_chunks,
            )
        )

        return GenerationRequest(
            request_id=request.request_id,
            task=request.task,
            query=request.query,
            context=sanitized_chunks,
            language=request.language,
            model_name=request.model_name,
            provider_name=request.provider_name,
            defense_enabled=request.defense_enabled,
            metadata={
                **request.metadata,
                "safe_context_applied": True,
                "safe_context_chunk_ids": [
                    item.chunk.chunk_id
                    for item in sanitized_chunks
                ],
                "safe_context_source_count": len(
                    sanitized_chunks
                ),
                "defense_decision": (
                    defense_result.final_decision.value
                ),
            },
        )

    @staticmethod
    def _retrieved_chunks_from_safe_context(
        context: BuiltContext,
        *,
        original_chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Reconstruct retrieved chunks from safe context."""

        original_by_id = {
            item.chunk.chunk_id: item
            for item in original_chunks
        }

        sanitized: list[RetrievedChunk] = []

        for source in context.sources:
            original = original_by_id.get(
                source.chunk_id
            )

            if original is not None:
                language = original.chunk.language
            else:
                language = (
                    ExperimentRunner._coerce_language(
                        source.language
                    )
                )

            source_metadata = dict(
                source.metadata
            )

            if original is not None:
                source_metadata = {
                    **original.chunk.metadata,
                    **source_metadata,
                }

            source_metadata[
                "sanitized_by_defense"
            ] = True

            if original is not None:
                start_line = (
                    original.chunk.start_line
                )
                end_line = original.chunk.end_line
                ast_node_type = (
                    original.chunk.ast_node_type
                )
                parent_symbol = (
                    original.chunk.parent_symbol
                )
            else:
                start_line = 1
                end_line = max(
                    1,
                    start_line
                    + len(
                        source.content.splitlines()
                    )
                    - 1,
                )
                ast_node_type = None
                parent_symbol = None

            chunk = CodeChunk(
                chunk_id=source.chunk_id,
                source_file_id=source.source_file_id,
                repository_id=source.repository_id,
                content=source.content,
                language=language,
                start_line=start_line,
                end_line=end_line,
                symbol_name=source.symbol_name,
                symbol_type=source.symbol_type,
                ast_node_type=ast_node_type,
                parent_symbol=parent_symbol,
                is_documentation=(
                    source.is_documentation
                ),
                trust=source.trust,
                metadata=source_metadata,
            )

            retrieval_metadata = dict(
                original.retrieval_metadata
                if original is not None
                else {}
            )

            retrieval_metadata.update(
                source.metadata
            )

            retrieval_metadata[
                "sanitized_by_defense"
            ] = True

            sanitized.append(
                RetrievedChunk(
                    chunk=chunk,
                    retrieval_score=(
                        source.retrieval_score
                    ),
                    rank=source.rank,
                    retriever_name=(
                        original.retriever_name
                        if original is not None
                        else "defense-sanitized"
                    ),
                    rerank_score=(
                        original.rerank_score
                        if original is not None
                        else None
                    ),
                    retrieval_metadata=(
                        retrieval_metadata
                    ),
                )
            )

        return sanitized

    @staticmethod
    def _coerce_language(
        language: Any,
    ) -> ProgrammingLanguage:
        """Convert a language value into ProgrammingLanguage."""

        if isinstance(
            language,
            ProgrammingLanguage,
        ):
            return language

        if isinstance(language, str):
            normalized = language.strip().lower()

            for candidate in ProgrammingLanguage:
                if normalized in {
                    candidate.value.lower(),
                    candidate.name.lower(),
                }:
                    return candidate

        return ProgrammingLanguage.UNKNOWN

    @staticmethod
    def _build_blocked_generation_result(
        *,
        request: ExperimentRunInput,
        config: ExperimentConfig,
        defense_result: DefensePipelineResult,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> GenerationResult:
        """Create a blocked generation result."""

        return GenerationResult(
            request_id=request.request_id,
            generated_code="",
            model_name=config.model_name,
            provider_name="defense-pipeline",
            status=GenerationStatus.BLOCKED,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=0.0,
            retrieved_chunk_ids=[
                item.chunk.chunk_id
                for item in retrieved_chunks
            ],
            metadata={
                "blocked_by_defense": True,
                "defense_decision": (
                    defense_result.final_decision.value
                ),
                "reason": (
                    "Retrieved context was blocked by "
                    "the unified defense pipeline."
                ),
            },
        )

    def _should_analyze_generated_code(
        self,
        generation_result: GenerationResult,
    ) -> bool:
        """Return whether generated code should undergo analysis."""

        if (
            self._defense_pipeline is None
            or not generation_result.generated_code.strip()
        ):
            return False

        defense_config = self._defense_pipeline.config

        if not defense_config.analyze_generated_code:
            return False

        return generation_result.status in {
            GenerationStatus.SUCCESS,
            GenerationStatus.VALIDATED,
        }

    @staticmethod
    def _detect_generation_change(
        generation_result: GenerationResult,
        baseline_generated_code: str | None,
    ) -> bool:
        """Compare generated output against a baseline."""

        if baseline_generated_code is None:
            return False

        return (
            generation_result.generated_code.strip()
            != baseline_generated_code.strip()
        )

    @staticmethod
    def _detect_vulnerability(
        findings: Sequence[SecurityFinding],
    ) -> bool:
        """Return whether a rejected security finding exists."""

        return any(
            finding.decision
            == SecurityDecision.REJECT
            for finding in findings
        )

    @staticmethod
    def _convert_security_findings(
        findings: Sequence[Any],
    ) -> tuple[SecurityFinding, ...]:
        """Convert security-layer findings to canonical findings."""

        converted: list[SecurityFinding] = []

        for finding in findings:
            location = getattr(
                finding,
                "location",
                None,
            )

            file_path = (
                getattr(
                    location,
                    "file_name",
                    None,
                )
                if location is not None
                else None
            )

            start_line = (
                getattr(
                    location,
                    "line_start",
                    None,
                )
                if location is not None
                else None
            )

            end_line = (
                getattr(
                    location,
                    "line_end",
                    None,
                )
                if location is not None
                else None
            )

            converted.append(
                SecurityFinding(
                    finding_id=str(
                        getattr(
                            finding,
                            "finding_id",
                            "unknown-finding",
                        )
                    ),
                    rule_id=str(
                        getattr(
                            finding,
                            "rule_id",
                            "unknown-rule",
                        )
                    ),
                    title=str(
                        getattr(
                            finding,
                            "title",
                            "Security finding",
                        )
                    ),
                    description=str(
                        getattr(
                            finding,
                            "description",
                            "",
                        )
                    ),
                    severity=finding.severity,
                    decision=finding.decision,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    evidence=str(
                        getattr(
                            finding,
                            "evidence",
                            "",
                        )
                    ),
                    analyzer=str(
                        getattr(
                            finding,
                            "source",
                            getattr(
                                finding,
                                "analyzer",
                                "defense-pipeline",
                            ),
                        )
                    ),
                    confidence=float(
                        getattr(
                            finding,
                            "confidence",
                            0.0,
                        )
                    ),
                    metadata={
                        **getattr(
                            finding,
                            "metadata",
                            {},
                        ),
                        "converted_from_security_layer": True,
                    },
                )
            )

        return tuple(converted)

    def metadata(self) -> dict[str, Any]:
        """Return non-sensitive runner metadata."""

        return {
            "runner": self.__class__.__name__,
            "retriever": self._retriever.name,
            "embedding_model": (
                self._retriever.embedding_model
            ),
            "generator_provider": (
                self._generator.provider_name
            ),
            "generator_model": (
                self._generator.model_name
            ),
            "defense_enabled": (
                self._defense_pipeline is not None
            ),
        }

    def __repr__(self) -> str:
        """Return a concise runner representation."""

        return (
            "ExperimentRunner("
            f"retriever={self._retriever.name!r}, "
            f"generator="
            f"{self._generator.provider_name!r}/"
            f"{self._generator.model_name!r}, "
            f"defense="
            f"{self._defense_pipeline is not None!r})"
        )


__all__ = [
    "ExperimentExecution",
    "ExperimentRunInput",
    "ExperimentRunner",
    "ExperimentRunnerConfigurationError",
    "ExperimentRunnerError",
    "ExperimentRunnerInputError",
]