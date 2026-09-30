"""Execution artifacts used to connect experiment execution and evaluation.

The experiment runner performs the actual experiment. This module provides a
small immutable container for the intermediate artifacts produced during that
execution so that the evaluation layer can calculate retrieval, generation,
and security metrics without executing the experiment again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models import (
    ExperimentCondition,
    ExperimentResult,
    RetrievedChunk,
)


class ExecutionArtifactsError(RuntimeError):
    """Base exception for execution artifact errors."""


class ExecutionArtifactsInputError(
    ExecutionArtifactsError,
    ValueError,
):
    """Raised when execution artifact input is invalid."""


@dataclass(frozen=True)
class ExecutionArtifacts:
    """All evaluation-relevant artifacts from one experiment execution."""

    experiment_id: str
    condition: ExperimentCondition
    query: str

    retrieved_chunks: tuple[RetrievedChunk, ...]
    relevant_chunk_ids: tuple[str, ...]

    generated_code: str
    reference_code: str

    defense_enabled: bool
    experiment_result: ExperimentResult

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate execution artifacts."""

        if not isinstance(self.experiment_id, str):
            raise ExecutionArtifactsInputError(
                "experiment_id must be a string."
            )

        if not self.experiment_id.strip():
            raise ExecutionArtifactsInputError(
                "experiment_id cannot be empty."
            )

        if not isinstance(self.condition, ExperimentCondition):
            raise ExecutionArtifactsInputError(
                "condition must be an ExperimentCondition."
            )

        if not isinstance(self.query, str):
            raise ExecutionArtifactsInputError(
                "query must be a string."
            )

        if not isinstance(self.retrieved_chunks, tuple):
            raise ExecutionArtifactsInputError(
                "retrieved_chunks must be a tuple."
            )

        for chunk in self.retrieved_chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise ExecutionArtifactsInputError(
                    "retrieved_chunks must contain only "
                    "RetrievedChunk instances."
                )

        if not isinstance(self.relevant_chunk_ids, tuple):
            raise ExecutionArtifactsInputError(
                "relevant_chunk_ids must be a tuple."
            )

        for chunk_id in self.relevant_chunk_ids:
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ExecutionArtifactsInputError(
                    "relevant_chunk_ids must contain non-empty strings."
                )

        if not isinstance(self.generated_code, str):
            raise ExecutionArtifactsInputError(
                "generated_code must be a string."
            )

        if not isinstance(self.reference_code, str):
            raise ExecutionArtifactsInputError(
                "reference_code must be a string."
            )

        if not isinstance(self.defense_enabled, bool):
            raise ExecutionArtifactsInputError(
                "defense_enabled must be a boolean."
            )

        if not isinstance(
            self.experiment_result,
            ExperimentResult,
        ):
            raise ExecutionArtifactsInputError(
                "experiment_result must be an ExperimentResult."
            )

        if not isinstance(self.metadata, dict):
            raise ExecutionArtifactsInputError(
                "metadata must be a dictionary."
            )

        if self.experiment_result.experiment_id != self.experiment_id:
            raise ExecutionArtifactsInputError(
                "experiment_result experiment_id does not match "
                "execution artifacts."
            )

        if self.experiment_result.condition != self.condition:
            raise ExecutionArtifactsInputError(
                "experiment_result condition does not match "
                "execution artifacts."
            )

    @property
    def retrieved_count(self) -> int:
        """Return the number of retrieved chunks."""

        return len(self.retrieved_chunks)

    @property
    def relevant_count(self) -> int:
        """Return the number of relevant chunk identifiers."""

        return len(self.relevant_chunk_ids)

    @property
    def output_nonempty(self) -> bool:
        """Return whether generated code contains meaningful output."""

        return bool(self.generated_code.strip())

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "experiment_id": self.experiment_id,
            "condition": self.condition.value,
            "query": self.query,
            "retrieved_chunks": [
                chunk.as_dict()
                for chunk in self.retrieved_chunks
            ],
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "generated_code": self.generated_code,
            "reference_code": self.reference_code,
            "defense_enabled": self.defense_enabled,
            "experiment_result": self.experiment_result.to_dict(),
            "retrieved_count": self.retrieved_count,
            "relevant_count": self.relevant_count,
            "output_nonempty": self.output_nonempty,
            "metadata": dict(self.metadata),
        }


def build_execution_artifacts(
    *,
    experiment_id: str,
    condition: ExperimentCondition,
    query: str,
    retrieved_chunks: tuple[RetrievedChunk, ...],
    relevant_chunk_ids: tuple[str, ...],
    generated_code: str,
    reference_code: str,
    defense_enabled: bool,
    experiment_result: ExperimentResult,
    metadata: dict[str, Any] | None = None,
) -> ExecutionArtifacts:
    """Construct validated execution artifacts."""

    return ExecutionArtifacts(
        experiment_id=experiment_id,
        condition=condition,
        query=query,
        retrieved_chunks=retrieved_chunks,
        relevant_chunk_ids=relevant_chunk_ids,
        generated_code=generated_code,
        reference_code=reference_code,
        defense_enabled=defense_enabled,
        experiment_result=experiment_result,
        metadata=dict(metadata or {}),
    )