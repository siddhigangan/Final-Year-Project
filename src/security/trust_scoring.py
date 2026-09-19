from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.models import CodeChunk, RetrievedChunk, SourceTrust


class TrustScoringError(RuntimeError):
    """Base exception for source trust-scoring failures."""


class TrustScoringInputError(TrustScoringError):
    """Raised when trust-scoring input is invalid."""


@dataclass(frozen=True)
class TrustScoringConfig:
    """Configuration for source trust scoring."""

    trusted_score: float = 1.00
    unknown_score: float = 0.60
    suspicious_score: float = 0.25
    poisoned_score: float = 0.00

    documentation_penalty: float = 0.05
    external_source_penalty: float = 0.10
    suspicious_metadata_penalty: float = 0.20

    minimum_allowed_score: float = 0.0
    maximum_allowed_score: float = 1.0

    suspicious_metadata_keys: tuple[str, ...] = (
        "poisoned",
        "suspicious",
        "untrusted",
        "injected",
        "malicious",
    )

    def __post_init__(self) -> None:
        scores = (
            self.trusted_score,
            self.unknown_score,
            self.suspicious_score,
            self.poisoned_score,
            self.documentation_penalty,
            self.external_source_penalty,
            self.suspicious_metadata_penalty,
        )

        if any(score < 0.0 for score in scores):
            raise TrustScoringInputError(
                "Trust scores and penalties cannot be negative"
            )

        if (
            self.minimum_allowed_score
            > self.maximum_allowed_score
        ):
            raise TrustScoringInputError(
                "minimum_allowed_score cannot exceed "
                "maximum_allowed_score"
            )

        if not (
            0.0
            <= self.minimum_allowed_score
            <= 1.0
        ):
            raise TrustScoringInputError(
                "minimum_allowed_score must be between 0 and 1"
            )

        if not (
            0.0
            <= self.maximum_allowed_score
            <= 1.0
        ):
            raise TrustScoringInputError(
                "maximum_allowed_score must be between 0 and 1"
            )

        for key in self.suspicious_metadata_keys:
            if not isinstance(key, str) or not key.strip():
                raise TrustScoringInputError(
                    "Suspicious metadata keys must be "
                    "non-empty strings"
                )


@dataclass(frozen=True)
class TrustAssessment:
    """Trust assessment for one retrieved source."""

    chunk_id: str
    source_file_id: str
    repository_id: str
    trust: SourceTrust
    score: float
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trusted(self) -> bool:
        """Return whether the source has full trust."""
        return self.trust == SourceTrust.TRUSTED

    @property
    def is_suspicious(self) -> bool:
        """Return whether the source is suspicious."""
        return self.trust == SourceTrust.SUSPICIOUS

    @property
    def is_poisoned(self) -> bool:
        """Return whether the source is explicitly poisoned."""
        return self.trust == SourceTrust.POISONED


@dataclass(frozen=True)
class TrustBatchAssessment:
    """Aggregated trust assessment for retrieved sources."""

    assessments: tuple[TrustAssessment, ...]
    average_score: float
    minimum_score: float
    maximum_score: float
    trusted_count: int
    unknown_count: int
    suspicious_count: int
    poisoned_count: int

    @property
    def total_count(self) -> int:
        """Return the number of assessed sources."""
        return len(self.assessments)


class SourceTrustScorer:
    """Calculate source trust scores for retrieved RAG context."""

    def __init__(
        self,
        config: TrustScoringConfig | None = None,
    ) -> None:
        self._config = config or TrustScoringConfig()

    @property
    def config(self) -> TrustScoringConfig:
        """Return the active trust-scoring configuration."""
        return self._config

    def score_chunk(
        self,
        chunk: CodeChunk,
    ) -> TrustAssessment:
        """Score a source chunk."""
        if not isinstance(chunk, CodeChunk):
            raise TrustScoringInputError(
                "chunk must be a CodeChunk"
            )

        return self._score_source(
            chunk_id=chunk.chunk_id,
            source_file_id=chunk.source_file_id,
            repository_id=chunk.repository_id,
            trust=chunk.trust,
            is_documentation=chunk.is_documentation,
            metadata=chunk.metadata,
        )

    def score_retrieved_chunk(
        self,
        retrieved: RetrievedChunk,
    ) -> TrustAssessment:
        """Score a retrieved chunk while preserving retrieval metadata."""
        if not isinstance(retrieved, RetrievedChunk):
            raise TrustScoringInputError(
                "retrieved must be a RetrievedChunk"
            )

        assessment = self.score_chunk(retrieved.chunk)

        metadata = {
            **assessment.metadata,
            "retrieval_score": retrieved.retrieval_score,
            "rank": retrieved.rank,
            "retriever_name": retrieved.retriever_name,
        }

        if retrieved.rerank_score is not None:
            metadata["rerank_score"] = retrieved.rerank_score

        return TrustAssessment(
            chunk_id=assessment.chunk_id,
            source_file_id=assessment.source_file_id,
            repository_id=assessment.repository_id,
            trust=assessment.trust,
            score=assessment.score,
            reasons=assessment.reasons,
            metadata=metadata,
        )

    def score_chunks(
        self,
        chunks: Iterable[CodeChunk],
    ) -> TrustBatchAssessment:
        """Score multiple source chunks."""
        normalized = tuple(chunks)

        assessments = tuple(
            self.score_chunk(chunk)
            for chunk in normalized
        )

        return self._aggregate(assessments)

    def score_retrieved_chunks(
        self,
        chunks: Iterable[RetrievedChunk],
    ) -> TrustBatchAssessment:
        """Score multiple retrieved chunks."""
        normalized = tuple(chunks)

        assessments = tuple(
            self.score_retrieved_chunk(chunk)
            for chunk in normalized
        )

        return self._aggregate(assessments)

    def _score_source(
        self,
        *,
        chunk_id: str,
        source_file_id: str,
        repository_id: str,
        trust: SourceTrust,
        is_documentation: bool,
        metadata: dict[str, Any],
    ) -> TrustAssessment:
        if not isinstance(trust, SourceTrust):
            raise TrustScoringInputError(
                "trust must be a SourceTrust value"
            )

        if not isinstance(metadata, dict):
            raise TrustScoringInputError(
                "metadata must be a dictionary"
            )

        score_mapping = {
            SourceTrust.TRUSTED: self._config.trusted_score,
            SourceTrust.UNKNOWN: self._config.unknown_score,
            SourceTrust.SUSPICIOUS: self._config.suspicious_score,
            SourceTrust.POISONED: self._config.poisoned_score,
        }

        score = score_mapping[trust]
        reasons: list[str] = [
            f"source_trust={trust.value}",
        ]

        if is_documentation:
            score -= self._config.documentation_penalty
            reasons.append("documentation_source")

        if self._is_external_source(metadata):
            score -= self._config.external_source_penalty
            reasons.append("external_source")

        suspicious_keys = self._find_suspicious_metadata(
            metadata
        )

        if suspicious_keys:
            score -= self._config.suspicious_metadata_penalty

            reasons.append(
                "suspicious_metadata:"
                + ",".join(suspicious_keys)
            )

        score = self._clamp(score)

        return TrustAssessment(
            chunk_id=chunk_id,
            source_file_id=source_file_id,
            repository_id=repository_id,
            trust=trust,
            score=score,
            reasons=tuple(reasons),
            metadata={
                "is_documentation": is_documentation,
                "suspicious_metadata_keys": suspicious_keys,
            },
        )

    def _is_external_source(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        return bool(
            metadata.get("external_source")
            or metadata.get("is_external")
            or metadata.get("external")
        )

    def _find_suspicious_metadata(
        self,
        metadata: dict[str, Any],
    ) -> tuple[str, ...]:
        matches: list[str] = []

        for key in self._config.suspicious_metadata_keys:
            value = metadata.get(key)

            if value is True:
                matches.append(key)
                continue

            if isinstance(value, str):
                normalized = value.strip().lower()

                if normalized in {
                    "true",
                    "yes",
                    "1",
                    "suspicious",
                    "poisoned",
                    "malicious",
                    "injected",
                }:
                    matches.append(key)

        return tuple(matches)

    def _clamp(self, score: float) -> float:
        return max(
            self._config.minimum_allowed_score,
            min(
                self._config.maximum_allowed_score,
                score,
            ),
        )

    @staticmethod
    def _aggregate(
        assessments: tuple[TrustAssessment, ...],
    ) -> TrustBatchAssessment:
        if not assessments:
            return TrustBatchAssessment(
                assessments=(),
                average_score=0.0,
                minimum_score=0.0,
                maximum_score=0.0,
                trusted_count=0,
                unknown_count=0,
                suspicious_count=0,
                poisoned_count=0,
            )

        scores = tuple(
            assessment.score
            for assessment in assessments
        )

        return TrustBatchAssessment(
            assessments=assessments,
            average_score=sum(scores) / len(scores),
            minimum_score=min(scores),
            maximum_score=max(scores),
            trusted_count=sum(
                assessment.trust == SourceTrust.TRUSTED
                for assessment in assessments
            ),
            unknown_count=sum(
                assessment.trust == SourceTrust.UNKNOWN
                for assessment in assessments
            ),
            suspicious_count=sum(
                assessment.trust == SourceTrust.SUSPICIOUS
                for assessment in assessments
            ),
            poisoned_count=sum(
                assessment.trust == SourceTrust.POISONED
                for assessment in assessments
            ),
        )


__all__ = [
    "SourceTrustScorer",
    "TrustAssessment",
    "TrustBatchAssessment",
    "TrustScoringConfig",
    "TrustScoringError",
    "TrustScoringInputError",
]