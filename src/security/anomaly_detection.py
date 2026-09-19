from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.models import CodeChunk, RetrievedChunk, SourceTrust


class AnomalyDetectionError(Exception):
    """Base exception for anomaly detection failures."""


class AnomalyDetectionInputError(AnomalyDetectionError):
    """Raised when anomaly detection receives invalid input."""


@dataclass(frozen=True)
class AnomalyDetectionConfig:
    """Configuration for deterministic anomaly detection."""

    anomaly_threshold: float = 0.50
    high_anomaly_threshold: float = 0.80

    suspicious_metadata_keys: tuple[str, ...] = (
        "poisoned",
        "suspicious",
        "untrusted",
        "injected",
        "malicious",
        "tampered",
    )

    suspicious_markers: tuple[str, ...] = (
        "ignore previous instructions",
        "ignore all previous instructions",
        "disregard previous instructions",
        "system prompt",
        "developer message",
        "do not follow",
        "override instructions",
        "hidden instruction",
        "secret instruction",
    )

    minimum_content_length: int = 1
    maximum_reasonable_content_length: int = 50000

    def __post_init__(self) -> None:
        if not 0.0 <= self.anomaly_threshold <= 1.0:
            raise ValueError("anomaly_threshold must be between 0 and 1")

        if not 0.0 <= self.high_anomaly_threshold <= 1.0:
            raise ValueError(
                "high_anomaly_threshold must be between 0 and 1"
            )

        if self.high_anomaly_threshold < self.anomaly_threshold:
            raise ValueError(
                "high_anomaly_threshold must be greater than or equal "
                "to anomaly_threshold"
            )

        if self.minimum_content_length < 0:
            raise ValueError(
                "minimum_content_length must be greater than or equal to 0"
            )

        if self.maximum_reasonable_content_length < (
            self.minimum_content_length
        ):
            raise ValueError(
                "maximum_reasonable_content_length must be greater than "
                "or equal to minimum_content_length"
            )

        if not self.suspicious_metadata_keys:
            raise ValueError("suspicious_metadata_keys cannot be empty")

        if not self.suspicious_markers:
            raise ValueError("suspicious_markers cannot be empty")


@dataclass(frozen=True)
class AnomalySignal:
    """A single deterministic anomaly signal."""

    signal_name: str
    score: float
    description: str
    evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal_name.strip():
            raise ValueError("signal_name cannot be empty")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between 0 and 1")

        if not self.description.strip():
            raise ValueError("description cannot be empty")


@dataclass(frozen=True)
class AnomalyAssessment:
    """Combined anomaly assessment for one source."""

    chunk_id: str
    source_file_id: str
    repository_id: str
    anomaly_score: float
    anomalous: bool
    high_risk: bool
    signals: tuple[AnomalySignal, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def signal_names(self) -> tuple[str, ...]:
        return tuple(signal.signal_name for signal in self.signals)


@dataclass(frozen=True)
class AnomalyBatchAssessment:
    """Aggregated anomaly assessment for multiple sources."""

    assessments: tuple[AnomalyAssessment, ...]
    average_score: float
    maximum_score: float
    anomalous_count: int
    high_risk_count: int

    @property
    def total_count(self) -> int:
        return len(self.assessments)

    @property
    def anomaly_rate(self) -> float:
        if self.total_count == 0:
            return 0.0

        return self.anomalous_count / self.total_count


class AnomalyDetector:
    """Deterministic detector for suspicious retrieval anomalies."""

    def __init__(
        self,
        config: AnomalyDetectionConfig | None = None,
    ) -> None:
        self._config = config or AnomalyDetectionConfig()

    @property
    def config(self) -> AnomalyDetectionConfig:
        return self._config

    def analyze_chunk(
        self,
        chunk: CodeChunk,
    ) -> AnomalyAssessment:
        """Analyze a CodeChunk for deterministic anomaly signals."""

        if not isinstance(chunk, CodeChunk):
            raise AnomalyDetectionInputError(
                "analyze_chunk requires a CodeChunk"
            )

        signals: list[AnomalySignal] = []

        self._append_metadata_signals(chunk, signals)
        self._append_trust_signals(chunk, signals)
        self._append_content_signals(chunk, signals)
        self._append_size_signals(chunk, signals)
        self._append_metadata_consistency_signals(chunk, signals)

        return self._build_assessment(chunk, signals)

    def analyze_retrieved_chunk(
        self,
        retrieved_chunk: RetrievedChunk,
    ) -> AnomalyAssessment:
        """Analyze a RetrievedChunk and its underlying source."""

        if not isinstance(retrieved_chunk, RetrievedChunk):
            raise AnomalyDetectionInputError(
                "analyze_retrieved_chunk requires a RetrievedChunk"
            )

        assessment = self.analyze_chunk(retrieved_chunk.chunk)

        metadata = dict(assessment.metadata)
        metadata["retrieval_score"] = retrieved_chunk.retrieval_score
        metadata["retrieval_rank"] = retrieved_chunk.rank
        metadata["retriever_name"] = retrieved_chunk.retriever_name

        return AnomalyAssessment(
            chunk_id=assessment.chunk_id,
            source_file_id=assessment.source_file_id,
            repository_id=assessment.repository_id,
            anomaly_score=assessment.anomaly_score,
            anomalous=assessment.anomalous,
            high_risk=assessment.high_risk,
            signals=assessment.signals,
            metadata=metadata,
        )

    def analyze_chunks(
        self,
        chunks: list[CodeChunk] | tuple[CodeChunk, ...],
    ) -> AnomalyBatchAssessment:
        """Analyze multiple code chunks."""

        if not isinstance(chunks, (list, tuple)):
            raise AnomalyDetectionInputError(
                "chunks must be a list or tuple of CodeChunk objects"
            )

        assessments = tuple(self.analyze_chunk(chunk) for chunk in chunks)
        return self._aggregate(assessments)

    def analyze_retrieved_chunks(
        self,
        chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
    ) -> AnomalyBatchAssessment:
        """Analyze multiple retrieved chunks."""

        if not isinstance(chunks, (list, tuple)):
            raise AnomalyDetectionInputError(
                "chunks must be a list or tuple of RetrievedChunk objects"
            )

        assessments = tuple(
            self.analyze_retrieved_chunk(chunk)
            for chunk in chunks
        )

        return self._aggregate(assessments)

    def _append_metadata_signals(
        self,
        chunk: CodeChunk,
        signals: list[AnomalySignal],
    ) -> None:
        normalized_keys = {
            str(key).lower()
            for key in chunk.metadata
        }

        suspicious_keys = {
            key.lower()
            for key in self._config.suspicious_metadata_keys
        }

        matched_keys = sorted(normalized_keys & suspicious_keys)

        for key in matched_keys:
            value = chunk.metadata.get(key)

            if value is None:
                for original_key, original_value in chunk.metadata.items():
                    if str(original_key).lower() == key:
                        value = original_value
                        break

            if value:
                signals.append(
                    AnomalySignal(
                        signal_name="suspicious_metadata",
                        score=0.80,
                        description=(
                            "Source metadata contains a suspicious "
                            "security-related marker."
                        ),
                        evidence=f"{key}={value!r}",
                        metadata={"key": key},
                    )
                )

    def _append_trust_signals(
        self,
        chunk: CodeChunk,
        signals: list[AnomalySignal],
    ) -> None:
        if chunk.trust == SourceTrust.POISONED:
            signals.append(
                AnomalySignal(
                    signal_name="poisoned_source",
                    score=1.0,
                    description="Source is explicitly marked as poisoned.",
                    evidence=chunk.trust.value,
                )
            )
        elif chunk.trust == SourceTrust.SUSPICIOUS:
            signals.append(
                AnomalySignal(
                    signal_name="suspicious_source",
                    score=0.75,
                    description=(
                        "Source is explicitly marked as suspicious."
                    ),
                    evidence=chunk.trust.value,
                )
            )
        elif chunk.trust == SourceTrust.UNKNOWN:
            signals.append(
                AnomalySignal(
                    signal_name="unknown_source",
                    score=0.30,
                    description=(
                        "Source trust has not been established."
                    ),
                    evidence=chunk.trust.value,
                )
            )

    def _append_content_signals(
        self,
        chunk: CodeChunk,
        signals: list[AnomalySignal],
    ) -> None:
        content = chunk.content

        if not content.strip():
            signals.append(
                AnomalySignal(
                    signal_name="empty_content",
                    score=0.40,
                    description="Source contains no meaningful content.",
                )
            )
            return

        lowered_content = content.lower()

        matched_markers = [
            marker
            for marker in self._config.suspicious_markers
            if marker.lower() in lowered_content
        ]

        if matched_markers:
            signals.append(
                AnomalySignal(
                    signal_name="instruction_like_content",
                    score=0.95,
                    description=(
                        "Source contains instruction-like language that "
                        "may attempt to influence the generation process."
                    ),
                    evidence="; ".join(matched_markers),
                    metadata={"matched_markers": matched_markers},
                )
            )

        control_patterns = (
            r"\bignore\s+(?:the\s+)?(?:previous|prior|above)\b",
            r"\bdisregard\s+(?:the\s+)?(?:previous|prior|above)\b",
            r"\boverride\s+(?:the\s+)?(?:system|developer|security)\b",
        )

        matched_patterns = [
            pattern
            for pattern in control_patterns
            if re.search(pattern, lowered_content)
        ]

        if matched_patterns and not matched_markers:
            signals.append(
                AnomalySignal(
                    signal_name="instruction_control_pattern",
                    score=0.90,
                    description=(
                        "Source contains a pattern associated with "
                        "instruction manipulation."
                    ),
                    evidence=matched_patterns[0],
                )
            )

    def _append_size_signals(
        self,
        chunk: CodeChunk,
        signals: list[AnomalySignal],
    ) -> None:
        content_length = len(chunk.content)

        if (
            content_length < self._config.minimum_content_length
            and self._config.minimum_content_length > 0
        ):
            signals.append(
                AnomalySignal(
                    signal_name="abnormally_short_content",
                    score=0.30,
                    description=(
                        "Source content is shorter than the configured "
                        "minimum."
                    ),
                    evidence=str(content_length),
                )
            )

        if content_length > self._config.maximum_reasonable_content_length:
            signals.append(
                AnomalySignal(
                    signal_name="abnormally_large_content",
                    score=0.55,
                    description=(
                        "Source content exceeds the configured reasonable "
                        "size."
                    ),
                    evidence=str(content_length),
                )
            )

    def _append_metadata_consistency_signals(
        self,
        chunk: CodeChunk,
        signals: list[AnomalySignal],
    ) -> None:
        metadata = chunk.metadata

        declared_language = metadata.get("language")

        if declared_language is not None:
            declared = str(declared_language).strip().lower()
            actual = chunk.language.value.lower()

            if declared and declared != actual:
                signals.append(
                    AnomalySignal(
                        signal_name="language_metadata_mismatch",
                        score=0.65,
                        description=(
                            "Metadata language does not match the actual "
                            "chunk language."
                        ),
                        evidence=f"declared={declared}, actual={actual}",
                    )
                )

        declared_repository = metadata.get("repository_id")

        if (
            declared_repository is not None
            and str(declared_repository) != chunk.repository_id
        ):
            signals.append(
                AnomalySignal(
                    signal_name="repository_metadata_mismatch",
                    score=0.60,
                    description=(
                        "Metadata repository identifier does not match "
                        "the chunk repository."
                    ),
                    evidence=(
                        f"declared={declared_repository}, "
                        f"actual={chunk.repository_id}"
                    ),
                )
            )

    def _build_assessment(
        self,
        chunk: CodeChunk,
        signals: list[AnomalySignal],
    ) -> AnomalyAssessment:
        anomaly_score = self._calculate_score(signals)

        anomalous = anomaly_score >= self._config.anomaly_threshold
        high_risk = anomaly_score >= self._config.high_anomaly_threshold

        return AnomalyAssessment(
            chunk_id=chunk.chunk_id,
            source_file_id=chunk.source_file_id,
            repository_id=chunk.repository_id,
            anomaly_score=anomaly_score,
            anomalous=anomalous,
            high_risk=high_risk,
            signals=tuple(signals),
            metadata={
                "trust": chunk.trust.value,
                "language": chunk.language.value,
                "is_documentation": chunk.is_documentation,
            },
        )

    def _calculate_score(
        self,
        signals: list[AnomalySignal],
    ) -> float:
        if not signals:
            return 0.0

        # Use the strongest signal rather than blindly summing scores.
        # This prevents many low-value signals from overwhelming one
        # assessment and keeps the result in [0, 1].
        return max(signal.score for signal in signals)

    def _aggregate(
        self,
        assessments: tuple[AnomalyAssessment, ...],
    ) -> AnomalyBatchAssessment:
        if not assessments:
            return AnomalyBatchAssessment(
                assessments=(),
                average_score=0.0,
                maximum_score=0.0,
                anomalous_count=0,
                high_risk_count=0,
            )

        scores = tuple(
            assessment.anomaly_score
            for assessment in assessments
        )

        anomalous_count = sum(
            assessment.anomalous
            for assessment in assessments
        )

        high_risk_count = sum(
            assessment.high_risk
            for assessment in assessments
        )

        return AnomalyBatchAssessment(
            assessments=assessments,
            average_score=sum(scores) / len(scores),
            maximum_score=max(scores),
            anomalous_count=anomalous_count,
            high_risk_count=high_risk_count,
        )