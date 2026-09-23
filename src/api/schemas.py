"""Request and response schemas for the SecureCodeRAG API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class APISchemaError(ValueError):
    """Raised when an API schema contains invalid data."""


@dataclass(frozen=True, slots=True)
class HealthResponse:
    """Health information returned by the API."""

    status: str = "ok"
    service: str = "SecureCodeRAG"
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible representation."""
        return {
            "status": self.status,
            "service": self.service,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class RepositoryIngestRequest:
    """Request to ingest an authorized local repository."""

    repository_path: str
    include_content: bool = True

    def __post_init__(self) -> None:
        """Validate the ingestion request."""
        if not isinstance(self.repository_path, str):
            raise APISchemaError("repository_path must be a string.")

        if not self.repository_path.strip():
            raise APISchemaError(
                "repository_path cannot be empty."
            )

        if not isinstance(self.include_content, bool):
            raise APISchemaError(
                "include_content must be a boolean."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "repository_path": self.repository_path,
            "include_content": self.include_content,
        }


@dataclass(frozen=True, slots=True)
class RepositoryIngestResponse:
    """Result of repository ingestion."""

    repository_id: str
    root_path: str
    file_count: int
    ingested_file_count: int
    skipped_file_count: int
    failed_file_count: int
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the ingestion response."""
        if not self.repository_id.strip():
            raise APISchemaError(
                "repository_id cannot be empty."
            )

        if not self.root_path.strip():
            raise APISchemaError(
                "root_path cannot be empty."
            )

        for field_name in (
            "file_count",
            "ingested_file_count",
            "skipped_file_count",
            "failed_file_count",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, int):
                raise APISchemaError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise APISchemaError(
                    f"{field_name} cannot be negative."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "repository_id": self.repository_id,
            "root_path": self.root_path,
            "file_count": self.file_count,
            "ingested_file_count": self.ingested_file_count,
            "skipped_file_count": self.skipped_file_count,
            "failed_file_count": self.failed_file_count,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Request to retrieve relevant code from the repository."""

    query: str
    top_k: int = 5
    repository_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the search request."""
        if not isinstance(self.query, str):
            raise APISchemaError("query must be a string.")

        if not self.query.strip():
            raise APISchemaError("query cannot be empty.")

        if not isinstance(self.top_k, int):
            raise APISchemaError("top_k must be an integer.")

        if self.top_k < 1:
            raise APISchemaError("top_k must be at least 1.")

        if self.top_k > 100:
            raise APISchemaError(
                "top_k cannot be greater than 100."
            )

        if self.repository_id is not None:
            if not isinstance(self.repository_id, str):
                raise APISchemaError(
                    "repository_id must be a string or None."
                )

            if not self.repository_id.strip():
                raise APISchemaError(
                    "repository_id cannot be empty."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "repository_id": self.repository_id,
        }


@dataclass(frozen=True, slots=True)
class SearchResultItem:
    """Serializable representation of a retrieved code chunk."""

    chunk_id: str
    source_file_id: str
    repository_id: str
    content: str
    language: str
    start_line: int
    end_line: int
    retrieval_score: float
    rank: int
    trust: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "chunk_id": self.chunk_id,
            "source_file_id": self.source_file_id,
            "repository_id": self.repository_id,
            "content": self.content,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "retrieval_score": self.retrieval_score,
            "rank": self.rank,
            "trust": self.trust,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Response containing retrieved code chunks."""

    query: str
    results: tuple[SearchResultItem, ...]
    requested_top_k: int
    result_count: int
    retriever_name: str
    embedding_model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "query": self.query,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "requested_top_k": self.requested_top_k,
            "result_count": self.result_count,
            "retriever_name": self.retriever_name,
            "embedding_model": self.embedding_model,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SecurityAnalysisRequest:
    """Request to analyze retrieved context for security risks."""

    query: str
    context: str
    repository_id: str | None = None

    def __post_init__(self) -> None:
        """Validate the security analysis request."""
        if not isinstance(self.query, str):
            raise APISchemaError("query must be a string.")

        if not self.query.strip():
            raise APISchemaError("query cannot be empty.")

        if not isinstance(self.context, str):
            raise APISchemaError("context must be a string.")

        if not self.context.strip():
            raise APISchemaError("context cannot be empty.")

        if self.repository_id is not None:
            if not isinstance(self.repository_id, str):
                raise APISchemaError(
                    "repository_id must be a string or None."
                )

            if not self.repository_id.strip():
                raise APISchemaError(
                    "repository_id cannot be empty."
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "query": self.query,
            "context": self.context,
            "repository_id": self.repository_id,
        }


@dataclass(frozen=True, slots=True)
class SecurityAnalysisResponse:
    """Security analysis result returned by the API."""

    decision: str
    risk_level: str
    finding_count: int
    context_safe_to_use: bool
    findings: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the security analysis response."""
        if not self.decision.strip():
            raise APISchemaError("decision cannot be empty.")

        if not self.risk_level.strip():
            raise APISchemaError("risk_level cannot be empty.")

        if not isinstance(self.finding_count, int):
            raise APISchemaError(
                "finding_count must be an integer."
            )

        if self.finding_count < 0:
            raise APISchemaError(
                "finding_count cannot be negative."
            )

        if not isinstance(self.context_safe_to_use, bool):
            raise APISchemaError(
                "context_safe_to_use must be a boolean."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "decision": self.decision,
            "risk_level": self.risk_level,
            "finding_count": self.finding_count,
            "context_safe_to_use": self.context_safe_to_use,
            "findings": [
                dict(finding)
                for finding in self.findings
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class APIErrorResponse:
    """Standard API error response."""

    error: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the API error."""
        if not self.error.strip():
            raise APISchemaError("error cannot be empty.")

        if not self.message.strip():
            raise APISchemaError("message cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "error": self.error,
            "message": self.message,
            "details": dict(self.details),
        }