from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models import ProgrammingLanguage, RetrievedChunk
from src.retrieval.context_builder import BuiltContext


class ContextValidationError(Exception):
    """Base exception for context validation failures."""


class ContextValidationInputError(ContextValidationError):
    """Raised when context validation receives invalid input."""


@dataclass(frozen=True)
class ContextValidationConfig:
    """Configuration for retrieval-context validation."""

    max_sources: int = 20
    max_context_characters: int = 50000
    max_duplicate_chunk_ratio: float = 0.50

    require_repository_consistency: bool = True
    require_language_consistency: bool = False
    require_provenance: bool = True

    reject_missing_chunk_ids: bool = True
    reject_missing_source_file_ids: bool = True
    reject_missing_repository_ids: bool = True

    def __post_init__(self) -> None:
        if self.max_sources <= 0:
            raise ValueError("max_sources must be greater than zero")

        if self.max_context_characters <= 0:
            raise ValueError(
                "max_context_characters must be greater than zero"
            )

        if not 0.0 <= self.max_duplicate_chunk_ratio <= 1.0:
            raise ValueError(
                "max_duplicate_chunk_ratio must be between 0 and 1"
            )


@dataclass(frozen=True)
class ContextValidationIssue:
    """A single context-validation issue."""

    issue_type: str
    severity: str
    description: str
    evidence: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.issue_type.strip():
            raise ValueError("issue_type cannot be empty")

        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(
                "severity must be info, warning, or error"
            )

        if not self.description.strip():
            raise ValueError("description cannot be empty")


@dataclass(frozen=True)
class ContextValidationResult:
    """Result of validating a retrieval context."""

    valid: bool
    safe_to_use: bool
    source_count: int
    unique_source_count: int
    duplicate_count: int
    context_characters: int
    repositories: tuple[str, ...]
    languages: tuple[str, ...]
    issues: tuple[ContextValidationIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def error_count(self) -> int:
        return sum(
            issue.severity == "error"
            for issue in self.issues
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity == "warning"
            for issue in self.issues
        )

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0


class ContextValidator:
    """Validates retrieval context before generation."""

    def __init__(
        self,
        config: ContextValidationConfig | None = None,
    ) -> None:
        self._config = config or ContextValidationConfig()

    @property
    def config(self) -> ContextValidationConfig:
        return self._config

    def validate(
        self,
        context: BuiltContext,
        retrieved_chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
    ) -> ContextValidationResult:
        """
        Validate a BuiltContext against its retrieved source chunks.
        """

        if not isinstance(context, BuiltContext):
            raise ContextValidationInputError(
                "context must be a BuiltContext"
            )

        if not isinstance(retrieved_chunks, (list, tuple)):
            raise ContextValidationInputError(
                "retrieved_chunks must be a list or tuple"
            )

        for chunk in retrieved_chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise ContextValidationInputError(
                    "all retrieved_chunks must be RetrievedChunk objects"
                )

        issues: list[ContextValidationIssue] = []

        source_count = len(retrieved_chunks)

        if source_count > self._config.max_sources:
            issues.append(
                ContextValidationIssue(
                    issue_type="source_count_exceeded",
                    severity="error",
                    description=(
                        "The number of retrieved sources exceeds "
                        "the configured context limit."
                    ),
                    evidence=str(source_count),
                )
            )

        self._validate_provenance(
            retrieved_chunks,
            issues,
        )

        self._validate_duplicates(
            retrieved_chunks,
            issues,
        )

        self._validate_repositories(
            retrieved_chunks,
            issues,
        )

        self._validate_languages(
            retrieved_chunks,
            issues,
        )

        self._validate_context_size(
            context,
            issues,
        )

        self._validate_context_sources(
            context,
            retrieved_chunks,
            issues,
        )

        repositories = tuple(
            sorted(
                {
                    chunk.chunk.repository_id
                    for chunk in retrieved_chunks
                    if chunk.chunk.repository_id.strip()
                }
            )
        )

        languages = tuple(
            sorted(
                {
                    chunk.chunk.language.value
                    for chunk in retrieved_chunks
                }
            )
        )

        unique_source_count = len(
            {
                chunk.chunk.chunk_id
                for chunk in retrieved_chunks
                if chunk.chunk.chunk_id.strip()
            }
        )

        duplicate_count = max(
            0,
            source_count - unique_source_count,
        )

        valid = not any(
            issue.severity == "error"
            for issue in issues
        )

        safe_to_use = valid

        return ContextValidationResult(
            valid=valid,
            safe_to_use=safe_to_use,
            source_count=source_count,
            unique_source_count=unique_source_count,
            duplicate_count=duplicate_count,
            context_characters=len(context.text),
            repositories=repositories,
            languages=languages,
            issues=tuple(issues),
            metadata={
                "truncated": context.truncated,
                "context_query": context.query,
            },
        )

    def validate_chunks(
        self,
        retrieved_chunks: list[RetrievedChunk]
        | tuple[RetrievedChunk, ...],
    ) -> ContextValidationResult:
        """
        Validate retrieved chunks without requiring a pre-built context.

        A temporary context is constructed from the source content so that
        the same validation rules can be applied.
        """

        if not isinstance(retrieved_chunks, (list, tuple)):
            raise ContextValidationInputError(
                "retrieved_chunks must be a list or tuple"
            )

        text = "\n\n".join(
            chunk.chunk.content
            for chunk in retrieved_chunks
            if isinstance(chunk, RetrievedChunk)
        )

        context = BuiltContext(
            text=text,
            sources=(),
            query=None,
            total_characters=len(text),
            total_lines=text.count("\n") + 1 if text else 0,
            truncated=False,
            metadata={},
        )

        return self.validate(
            context,
            retrieved_chunks,
        )

    def _validate_provenance(
        self,
        chunks: tuple[RetrievedChunk, ...]
        | list[RetrievedChunk],
        issues: list[ContextValidationIssue],
    ) -> None:
        if not self._config.require_provenance:
            return

        for index, retrieved in enumerate(chunks):
            chunk = retrieved.chunk
            location = f"retrieved_chunks[{index}]"

            if (
                self._config.reject_missing_chunk_ids
                and not chunk.chunk_id.strip()
            ):
                issues.append(
                    ContextValidationIssue(
                        issue_type="missing_chunk_id",
                        severity="error",
                        description=(
                            "Retrieved source is missing its chunk "
                            "identifier."
                        ),
                        evidence=location,
                    )
                )

            if (
                self._config.reject_missing_source_file_ids
                and not chunk.source_file_id.strip()
            ):
                issues.append(
                    ContextValidationIssue(
                        issue_type="missing_source_file_id",
                        severity="error",
                        description=(
                            "Retrieved source is missing its source "
                            "file identifier."
                        ),
                        evidence=location,
                    )
                )

            if (
                self._config.reject_missing_repository_ids
                and not chunk.repository_id.strip()
            ):
                issues.append(
                    ContextValidationIssue(
                        issue_type="missing_repository_id",
                        severity="error",
                        description=(
                            "Retrieved source is missing its repository "
                            "identifier."
                        ),
                        evidence=location,
                    )
                )

    def _validate_duplicates(
        self,
        chunks: tuple[RetrievedChunk, ...]
        | list[RetrievedChunk],
        issues: list[ContextValidationIssue],
    ) -> None:
        if not chunks:
            return

        identifiers = [
            chunk.chunk.chunk_id
            for chunk in chunks
            if chunk.chunk.chunk_id.strip()
        ]

        duplicate_count = len(identifiers) - len(set(identifiers))

        if duplicate_count <= 0:
            return

        duplicate_ratio = duplicate_count / len(chunks)

        if duplicate_ratio > self._config.max_duplicate_chunk_ratio:
            severity = "error"
        else:
            severity = "warning"

        issues.append(
            ContextValidationIssue(
                issue_type="duplicate_sources",
                severity=severity,
                description=(
                    "The retrieval context contains duplicate chunk "
                    "identifiers."
                ),
                evidence=(
                    f"duplicates={duplicate_count}, "
                    f"ratio={duplicate_ratio:.3f}"
                ),
                metadata={
                    "duplicate_count": duplicate_count,
                    "duplicate_ratio": duplicate_ratio,
                },
            )
        )

    def _validate_repositories(
        self,
        chunks: tuple[RetrievedChunk, ...]
        | list[RetrievedChunk],
        issues: list[ContextValidationIssue],
    ) -> None:
        if not self._config.require_repository_consistency:
            return

        repositories = {
            chunk.chunk.repository_id
            for chunk in chunks
            if chunk.chunk.repository_id.strip()
        }

        if len(repositories) <= 1:
            return

        issues.append(
            ContextValidationIssue(
                issue_type="repository_inconsistency",
                severity="error",
                description=(
                    "Retrieved sources originate from multiple "
                    "repositories."
                ),
                evidence=", ".join(sorted(repositories)),
                metadata={
                    "repositories": sorted(repositories),
                },
            )
        )

    def _validate_languages(
        self,
        chunks: tuple[RetrievedChunk, ...]
        | list[RetrievedChunk],
        issues: list[ContextValidationIssue],
    ) -> None:
        if not self._config.require_language_consistency:
            return

        languages = {
            chunk.chunk.language
            for chunk in chunks
            if chunk.chunk.language != ProgrammingLanguage.UNKNOWN
        }

        if len(languages) <= 1:
            return

        language_values = sorted(
            language.value
            for language in languages
        )

        issues.append(
            ContextValidationIssue(
                issue_type="language_inconsistency",
                severity="warning",
                description=(
                    "Retrieved sources contain multiple programming "
                    "languages."
                ),
                evidence=", ".join(language_values),
                metadata={
                    "languages": language_values,
                },
            )
        )

    def _validate_context_size(
        self,
        context: BuiltContext,
        issues: list[ContextValidationIssue],
    ) -> None:
        if len(context.text) <= self._config.max_context_characters:
            return

        issues.append(
            ContextValidationIssue(
                issue_type="context_size_exceeded",
                severity="error",
                description=(
                    "Built context exceeds the configured character "
                    "limit."
                ),
                evidence=str(len(context.text)),
            )
        )

    def _validate_context_sources(
        self,
        context: BuiltContext,
        chunks: tuple[RetrievedChunk, ...]
        | list[RetrievedChunk],
        issues: list[ContextValidationIssue],
    ) -> None:
        if not context.sources:
            return

        context_chunk_ids = {
            source.chunk_id
            for source in context.sources
            if source.chunk_id.strip()
        }

        retrieved_chunk_ids = {
            chunk.chunk.chunk_id
            for chunk in chunks
            if chunk.chunk.chunk_id.strip()
        }

        missing_from_context = (
            retrieved_chunk_ids - context_chunk_ids
        )

        if missing_from_context:
            issues.append(
                ContextValidationIssue(
                    issue_type="source_provenance_mismatch",
                    severity="warning",
                    description=(
                        "Some retrieved chunks are not represented "
                        "in the built context provenance."
                    ),
                    evidence=", ".join(
                        sorted(missing_from_context)
                    ),
                    metadata={
                        "missing_chunk_ids": sorted(
                            missing_from_context
                        ),
                    },
                )
            )