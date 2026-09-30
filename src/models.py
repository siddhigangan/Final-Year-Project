"""Shared domain models for the SecureCodeRAG research pipeline.

The models in this module define stable contracts between the major
pipeline components:

    ingestion -> parsing -> chunking -> retrieval -> generation
    -> security analysis -> evaluation

The module intentionally depends only on the Python standard library so
that the project's domain layer remains lightweight and portable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ModelValidationError(ValueError):
    """Raised when a SecureCodeRAG domain model receives invalid data."""


class ProgrammingLanguage(str, Enum):
    """Programming languages supported by the initial pipeline."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    UNKNOWN = "unknown"


class SourceTrust(str, Enum):
    """Trust classification assigned to an ingested source."""

    TRUSTED = "trusted"
    UNKNOWN = "unknown"
    SUSPICIOUS = "suspicious"
    POISONED = "poisoned"


class GenerationStatus(str, Enum):
    """Lifecycle status of a code-generation request."""

    SUCCESS = "success"
    BLOCKED = "blocked"
    FAILED = "failed"
    VALIDATED = "validated"


class SecuritySeverity(str, Enum):
    """Severity levels used by security analysis."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityDecision(str, Enum):
    """Decision produced by the output-security decision engine."""

    PASS = "pass"
    FLAG = "flag"
    REJECT = "reject"


class ExperimentCondition(str, Enum):
    """Controlled experimental conditions defined by the research design."""

    CLEAN_NO_DEFENSE = "A_clean_no_defense"
    POISONED_NO_DEFENSE = "B_poisoned_no_defense"
    POISONED_DEFENSE = "C_poisoned_defense"
    CLEAN_DEFENSE = "D_clean_defense"


@dataclass(slots=True)
class SourceFile:
    """Represents a source file discovered during repository ingestion.

    A SourceFile contains repository-level metadata but does not contain
    chunking or retrieval information. Those concerns belong to later
    pipeline stages.
    """

    file_id: str
    repository_id: str
    relative_path: str
    language: ProgrammingLanguage
    content: str
    size_bytes: int
    sha256: str
    trust: SourceTrust = SourceTrust.TRUSTED
    is_documentation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.file_id.strip():
            raise ModelValidationError("file_id cannot be empty.")

        if not self.repository_id.strip():
            raise ModelValidationError("repository_id cannot be empty.")

        if not self.relative_path.strip():
            raise ModelValidationError("relative_path cannot be empty.")

        if not isinstance(self.language, ProgrammingLanguage):
            raise ModelValidationError(
                "language must be a ProgrammingLanguage value."
            )

        if self.size_bytes < 0:
            raise ModelValidationError("size_bytes cannot be negative.")

        if not self.sha256.strip():
            raise ModelValidationError("sha256 cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class CodeChunk:
    """Represents an AST-aware or document-aware repository chunk."""

    chunk_id: str
    source_file_id: str
    repository_id: str
    content: str
    language: ProgrammingLanguage
    start_line: int
    end_line: int
    symbol_name: str | None = None
    symbol_type: str | None = None
    ast_node_type: str | None = None
    parent_symbol: str | None = None
    is_documentation: bool = False
    trust: SourceTrust = SourceTrust.TRUSTED
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise ModelValidationError("chunk_id cannot be empty.")

        if not self.source_file_id.strip():
            raise ModelValidationError("source_file_id cannot be empty.")

        if not self.repository_id.strip():
            raise ModelValidationError("repository_id cannot be empty.")

        if not self.content.strip():
            raise ModelValidationError("content cannot be empty.")

        if self.start_line < 1:
            raise ModelValidationError("start_line must be >= 1.")

        if self.end_line < self.start_line:
            raise ModelValidationError(
                "end_line must be greater than or equal to start_line."
            )

        if not isinstance(self.language, ProgrammingLanguage):
            raise ModelValidationError(
                "language must be a ProgrammingLanguage value."
            )

    @property
    def line_count(self) -> int:
        """Return the number of source lines represented by the chunk."""
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class RetrievedChunk:
    """Represents a chunk returned by the retrieval subsystem."""

    chunk: CodeChunk
    retrieval_score: float
    rank: int
    retriever_name: str = "default"
    rerank_score: float | None = None
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, CodeChunk):
            raise ModelValidationError("chunk must be a CodeChunk.")

        if self.rank < 1:
            raise ModelValidationError("rank must be >= 1.")

        if not self.retriever_name.strip():
            raise ModelValidationError("retriever_name cannot be empty.")

        if self.rerank_score is not None and not isinstance(
            self.rerank_score, (int, float)
        ):
            raise ModelValidationError(
                "rerank_score must be numeric or None."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class GenerationRequest:
    """Input contract for a code-generation operation."""

    request_id: str
    task: str
    query: str
    context: list[RetrievedChunk] = field(default_factory=list)
    language: ProgrammingLanguage = ProgrammingLanguage.UNKNOWN
    model_name: str = ""
    provider_name: str = ""
    defense_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ModelValidationError("request_id cannot be empty.")

        if not self.task.strip():
            raise ModelValidationError("task cannot be empty.")

        if not self.query.strip():
            raise ModelValidationError("query cannot be empty.")

        if not isinstance(self.language, ProgrammingLanguage):
            raise ModelValidationError(
                "language must be a ProgrammingLanguage value."
            )

        invalid_context = [
            item
            for item in self.context
            if not isinstance(item, RetrievedChunk)
        ]

        if invalid_context:
            raise ModelValidationError(
                "Every context item must be a RetrievedChunk."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    """Output produced by a code-generation provider."""

    request_id: str
    generated_code: str
    model_name: str
    provider_name: str
    status: GenerationStatus = GenerationStatus.SUCCESS
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ModelValidationError("request_id cannot be empty.")

        if not self.model_name.strip():
            raise ModelValidationError("model_name cannot be empty.")

        if not self.provider_name.strip():
            raise ModelValidationError("provider_name cannot be empty.")

        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ModelValidationError("prompt_tokens cannot be negative.")

        if (
            self.completion_tokens is not None
            and self.completion_tokens < 0
        ):
            raise ModelValidationError(
                "completion_tokens cannot be negative."
            )

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ModelValidationError("latency_ms cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class SecurityFinding:
    """Represents a finding from static or security analysis."""

    finding_id: str
    rule_id: str
    title: str
    description: str
    severity: SecuritySeverity
    decision: SecurityDecision
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    evidence: str | None = None
    analyzer: str = ""
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.finding_id.strip():
            raise ModelValidationError("finding_id cannot be empty.")

        if not self.rule_id.strip():
            raise ModelValidationError("rule_id cannot be empty.")

        if not self.title.strip():
            raise ModelValidationError("title cannot be empty.")

        if not self.description.strip():
            raise ModelValidationError("description cannot be empty.")

        if not isinstance(self.severity, SecuritySeverity):
            raise ModelValidationError(
                "severity must be a SecuritySeverity value."
            )

        if not isinstance(self.decision, SecurityDecision):
            raise ModelValidationError(
                "decision must be a SecurityDecision value."
            )

        if self.start_line is not None and self.start_line < 1:
            raise ModelValidationError("start_line must be >= 1.")

        if self.end_line is not None and self.end_line < 1:
            raise ModelValidationError("end_line must be >= 1.")

        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ModelValidationError(
                "end_line must be >= start_line."
            )

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ModelValidationError(
                "confidence must be between 0.0 and 1.0."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for one controlled SecureCodeRAG experiment."""

    experiment_id: str
    condition: ExperimentCondition
    repository_id: str
    task_type: str
    language: ProgrammingLanguage
    model_name: str
    retriever_name: str
    top_k: int = 5
    defense_layers: list[str] = field(default_factory=list)
    seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ModelValidationError("experiment_id cannot be empty.")

        if not self.repository_id.strip():
            raise ModelValidationError("repository_id cannot be empty.")

        if not self.task_type.strip():
            raise ModelValidationError("task_type cannot be empty.")

        if not self.model_name.strip():
            raise ModelValidationError("model_name cannot be empty.")

        if not self.retriever_name.strip():
            raise ModelValidationError("retriever_name cannot be empty.")

        if not isinstance(self.condition, ExperimentCondition):
            raise ModelValidationError(
                "condition must be an ExperimentCondition value."
            )

        if not isinstance(self.language, ProgrammingLanguage):
            raise ModelValidationError(
                "language must be a ProgrammingLanguage value."
            )

        if self.top_k < 1:
            raise ModelValidationError("top_k must be >= 1.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)


@dataclass(slots=True)
class ExperimentResult:
    """Aggregated result of one controlled experiment run.

    The causal-trace fields are intentionally explicit because the
    research design requires distinguishing:

        poison exists
        -> poison retrieved
        -> poison included in context
        -> generation changed
        -> vulnerability introduced
    """

    experiment_id: str
    condition: ExperimentCondition
    request_id: str
    poison_present: bool
    poison_retrieved: bool
    poison_in_context: bool
    generation_changed: bool
    vulnerability_introduced: bool
    poison_retrieval_rate: float | None = None
    attack_success_rate: float | None = None
    vulnerability_introduction_rate: float | None = None
    utility_score: float | None = None
    latency_ms: float | None = None
    findings: list[SecurityFinding] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ModelValidationError("experiment_id cannot be empty.")

        if not self.request_id.strip():
            raise ModelValidationError("request_id cannot be empty.")

        if not isinstance(self.condition, ExperimentCondition):
            raise ModelValidationError(
                "condition must be an ExperimentCondition value."
            )

        invalid_findings = [
            finding
            for finding in self.findings
            if not isinstance(finding, SecurityFinding)
        ]

        if invalid_findings:
            raise ModelValidationError(
                "Every finding must be a SecurityFinding."
            )

        metric_values = {
            "poison_retrieval_rate": self.poison_retrieval_rate,
            "attack_success_rate": self.attack_success_rate,
            "vulnerability_introduction_rate": (
                self.vulnerability_introduction_rate
            ),
            "utility_score": self.utility_score,
        }

        for metric_name, value in metric_values.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ModelValidationError(
                    f"{metric_name} must be between 0.0 and 1.0."
                )

        if self.latency_ms is not None and self.latency_ms < 0:
            raise ModelValidationError("latency_ms cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary representation."""
        return asdict(self)