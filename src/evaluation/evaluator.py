"""Evaluation integration for controlled Code-RAG experiments.

This module combines the existing retrieval, generation, and
security-utility metrics into one deterministic evaluation result.

The evaluator does not perform retrieval, generation, or defense itself.
It evaluates outputs that have already been produced by the experiment
runner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.generation_metrics import (
    GenerationMetrics,
    calculate_generation_metrics,
)
from src.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    calculate_retrieval_metrics,
)
from src.evaluation.security_utility import (
    SecurityUtilityObservation,
    calculate_security_utility_score,
)
from src.models import ExperimentResult, RetrievedChunk


class EvaluatorError(Exception):
    """Base exception for evaluation errors."""


class EvaluatorInputError(EvaluatorError, ValueError):
    """Raised when evaluator inputs are invalid."""


@dataclass(frozen=True)
class EvaluationInput:
    """Input required to evaluate one experiment result."""

    experiment_result: ExperimentResult

    retrieved_chunks: tuple[RetrievedChunk, ...] = ()

    relevant_chunk_ids: tuple[str, ...] = ()

    top_k: int = 5

    generated_code: str = ""
    reference_code: str = ""

    security_score: float = 0.0

    latency_ms: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.experiment_result,
            ExperimentResult,
        ):
            raise EvaluatorInputError(
                "experiment_result must be an ExperimentResult."
            )

        if not isinstance(self.retrieved_chunks, tuple):
            raise EvaluatorInputError(
                "retrieved_chunks must be a tuple."
            )

        if any(
            not isinstance(chunk, RetrievedChunk)
            for chunk in self.retrieved_chunks
        ):
            raise EvaluatorInputError(
                "retrieved_chunks must contain only RetrievedChunk "
                "instances."
            )

        if not isinstance(self.relevant_chunk_ids, tuple):
            raise EvaluatorInputError(
                "relevant_chunk_ids must be a tuple."
            )

        if any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in self.relevant_chunk_ids
        ):
            raise EvaluatorInputError(
                "relevant_chunk_ids must contain non-empty strings."
            )

        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise EvaluatorInputError(
                "top_k must be a positive integer."
            )

        if not isinstance(self.generated_code, str):
            raise EvaluatorInputError(
                "generated_code must be a string."
            )

        if not isinstance(self.reference_code, str):
            raise EvaluatorInputError(
                "reference_code must be a string."
            )

        if isinstance(self.security_score, bool):
            raise EvaluatorInputError(
                "security_score must be a number between 0.0 and 1.0."
            )

        if not isinstance(
            self.security_score,
            (int, float),
        ):
            raise EvaluatorInputError(
                "security_score must be a number between 0.0 and 1.0."
            )

        if not 0.0 <= float(self.security_score) <= 1.0:
            raise EvaluatorInputError(
                "security_score must be between 0.0 and 1.0."
            )

        if self.latency_ms is not None:
            if isinstance(self.latency_ms, bool):
                raise EvaluatorInputError(
                    "latency_ms must be a non-negative number or None."
                )

            if not isinstance(
                self.latency_ms,
                (int, float),
            ):
                raise EvaluatorInputError(
                    "latency_ms must be a non-negative number or None."
                )

            if self.latency_ms < 0:
                raise EvaluatorInputError(
                    "latency_ms must be non-negative."
                )

        if not isinstance(self.metadata, dict):
            raise EvaluatorInputError(
                "metadata must be a dictionary."
            )


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation output for one experiment."""

    experiment_id: str
    condition: str

    retrieval: RetrievalMetrics
    generation: GenerationMetrics

    security_score: float
    security_utility_score: float

    poison_present: bool
    poison_retrieved: bool
    poison_in_context: bool
    generation_changed: bool
    vulnerability_introduced: bool

    latency_ms: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise EvaluatorInputError(
                "experiment_id cannot be empty."
            )

        if not self.condition.strip():
            raise EvaluatorInputError(
                "condition cannot be empty."
            )

        if not isinstance(
            self.retrieval,
            RetrievalMetrics,
        ):
            raise EvaluatorInputError(
                "retrieval must be RetrievalMetrics."
            )

        if not isinstance(
            self.generation,
            GenerationMetrics,
        ):
            raise EvaluatorInputError(
                "generation must be GenerationMetrics."
            )

        _validate_normalized_score(
            self.security_score,
            "security_score",
        )

        _validate_normalized_score(
            self.security_utility_score,
            "security_utility_score",
        )

        if self.latency_ms is not None:
            if isinstance(self.latency_ms, bool):
                raise EvaluatorInputError(
                    "latency_ms must be non-negative."
                )

            if not isinstance(
                self.latency_ms,
                (int, float),
            ):
                raise EvaluatorInputError(
                    "latency_ms must be non-negative."
                )

            if self.latency_ms < 0:
                raise EvaluatorInputError(
                    "latency_ms must be non-negative."
                )

        if not isinstance(self.metadata, dict):
            raise EvaluatorInputError(
                "metadata must be a dictionary."
            )

    @property
    def retrieval_quality(self) -> float:
        """Return the primary retrieval-quality score."""
        return self.retrieval.ndcg_at_k

    @property
    def generation_quality(self) -> float:
        """Return the primary generation-quality score."""
        return self.generation.utility

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evaluation result."""
        return {
            "experiment_id": self.experiment_id,
            "condition": self.condition,
            "retrieval": self.retrieval.as_dict(),
            "generation": self.generation.as_dict(),
            "security_score": self.security_score,
            "security_utility_score": (
                self.security_utility_score
            ),
            "retrieval_quality": self.retrieval_quality,
            "generation_quality": self.generation_quality,
            "poison_present": self.poison_present,
            "poison_retrieved": self.poison_retrieved,
            "poison_in_context": self.poison_in_context,
            "generation_changed": self.generation_changed,
            "vulnerability_introduced": (
                self.vulnerability_introduced
            ),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


def _validate_normalized_score(
    value: float,
    name: str,
) -> None:
    """Validate a normalized score."""
    if isinstance(value, bool):
        raise EvaluatorInputError(
            f"{name} must be between 0.0 and 1.0."
        )

    if not isinstance(value, (int, float)):
        raise EvaluatorInputError(
            f"{name} must be between 0.0 and 1.0."
        )

    if not 0.0 <= float(value) <= 1.0:
        raise EvaluatorInputError(
            f"{name} must be between 0.0 and 1.0."
        )


def _materialize_chunks(
    chunks: Iterable[RetrievedChunk],
) -> tuple[RetrievedChunk, ...]:
    """Validate and materialize retrieved chunks."""
    if chunks is None:
        raise EvaluatorInputError(
            "retrieved_chunks cannot be None."
        )

    result = tuple(chunks)

    if any(
        not isinstance(chunk, RetrievedChunk)
        for chunk in result
    ):
        raise EvaluatorInputError(
            "retrieved_chunks must contain only RetrievedChunk "
            "instances."
        )

    return result


def calculate_evaluation(
    evaluation_input: EvaluationInput,
) -> EvaluationResult:
    """Calculate the complete evaluation for one experiment.

    Retrieval quality uses NDCG@K as the primary retrieval score.

    Generation quality uses the deterministic generation utility
    already defined by the generation metrics module.

    Security utility combines the supplied security score with
    retrieval and generation quality using the existing
    security-utility weighting.
    """
    if not isinstance(
        evaluation_input,
        EvaluationInput,
    ):
        raise EvaluatorInputError(
            "evaluation_input must be an EvaluationInput."
        )

    retrieved_chunks = _materialize_chunks(
        evaluation_input.retrieved_chunks
    )

    retrieval_metrics = calculate_retrieval_metrics(
        retrieved=retrieved_chunks,
        relevant_ids=evaluation_input.relevant_chunk_ids,
        k=evaluation_input.top_k,
        metadata={
            **evaluation_input.metadata,
            "experiment_id": (
                evaluation_input.experiment_result.experiment_id
            ),
        },
    )

    generation_metrics = calculate_generation_metrics(
        candidate_code=evaluation_input.generated_code,
        reference_code=evaluation_input.reference_code,
        metadata={
            **evaluation_input.metadata,
            "experiment_id": (
                evaluation_input.experiment_result.experiment_id
            ),
        },
    )

    retrieval_quality = retrieval_metrics.ndcg_at_k
    generation_quality = generation_metrics.utility

    security_utility_score = (
        calculate_security_utility_score(
            security_score=evaluation_input.security_score,
            retrieval_quality=retrieval_quality,
            generation_quality=generation_quality,
        )
    )

    experiment = evaluation_input.experiment_result

    observation = SecurityUtilityObservation(
        experiment_id=experiment.experiment_id,
        condition=experiment.condition.value,
        security_score=float(
            evaluation_input.security_score
        ),
        retrieval_quality=retrieval_quality,
        generation_quality=generation_quality,
        defense_enabled=bool(
            experiment.metadata.get(
                "defense_enabled",
                False,
            )
        ),
        blocked=bool(
            experiment.metadata.get(
                "defense_blocked",
                False,
            )
        ),
        attack_succeeded=bool(
            experiment.metadata.get(
                "attack_succeeded",
                experiment.vulnerability_introduced,
            )
        ),
        vulnerability_introduced=(
            experiment.vulnerability_introduced
        ),
        latency_ms=(
            evaluation_input.latency_ms
            if evaluation_input.latency_ms is not None
            else experiment.latency_ms
        ),
        metadata={
            **evaluation_input.metadata,
            "poison_present": experiment.poison_present,
            "poison_retrieved": experiment.poison_retrieved,
            "poison_in_context": experiment.poison_in_context,
            "generation_changed": experiment.generation_changed,
        },
    )

    metadata = {
        **experiment.metadata,
        **evaluation_input.metadata,
        "retrieval_quality_metric": "ndcg_at_k",
        "generation_quality_metric": "utility",
        "security_utility_weights": {
            "security": 0.50,
            "retrieval": 0.25,
            "generation": 0.25,
        },
        "security_utility_observation": observation.as_dict(),
    }

    return EvaluationResult(
        experiment_id=experiment.experiment_id,
        condition=experiment.condition.value,
        retrieval=retrieval_metrics,
        generation=generation_metrics,
        security_score=float(
            evaluation_input.security_score
        ),
        security_utility_score=security_utility_score,
        poison_present=experiment.poison_present,
        poison_retrieved=experiment.poison_retrieved,
        poison_in_context=experiment.poison_in_context,
        generation_changed=experiment.generation_changed,
        vulnerability_introduced=(
            experiment.vulnerability_introduced
        ),
        latency_ms=(
            evaluation_input.latency_ms
            if evaluation_input.latency_ms is not None
            else experiment.latency_ms
        ),
        metadata=metadata,
    )


def evaluate_experiment(
    experiment_result: ExperimentResult,
    *,
    retrieved_chunks: Iterable[RetrievedChunk],
    relevant_chunk_ids: Iterable[str] = (),
    top_k: int = 5,
    generated_code: str = "",
    reference_code: str = "",
    security_score: float = 0.0,
    latency_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Convenience wrapper for evaluating one experiment."""
    chunks = _materialize_chunks(retrieved_chunks)

    relevant_ids = tuple(relevant_chunk_ids)

    return calculate_evaluation(
        EvaluationInput(
            experiment_result=experiment_result,
            retrieved_chunks=chunks,
            relevant_chunk_ids=relevant_ids,
            top_k=top_k,
            generated_code=generated_code,
            reference_code=reference_code,
            security_score=security_score,
            latency_ms=latency_ms,
            metadata=dict(metadata or {}),
        )
    )