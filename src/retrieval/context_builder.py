"""Structured context construction for SecureCodeRAG retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models import RetrievedChunk


class ContextBuilderError(Exception):
    """Base exception for context-building failures."""


class ContextBuilderInputError(ContextBuilderError):
    """Raised when context-builder input is invalid."""


@dataclass(frozen=True)
class ContextSource:
    """Structured provenance information for one context source."""

    source_index: int
    chunk_id: str
    source_file_id: str
    repository_id: str
    relative_path: str
    language: str
    trust: str
    retrieval_score: float
    rank: int
    symbol_name: str | None
    symbol_type: str | None
    is_documentation: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_index <= 0:
            raise ContextBuilderInputError(
                "source_index must be greater than zero."
            )

        if not self.chunk_id.strip():
            raise ContextBuilderInputError(
                "chunk_id cannot be empty."
            )

        if not self.source_file_id.strip():
            raise ContextBuilderInputError(
                "source_file_id cannot be empty."
            )

        if not self.repository_id.strip():
            raise ContextBuilderInputError(
                "repository_id cannot be empty."
            )

        if not self.relative_path.strip():
            raise ContextBuilderInputError(
                "relative_path cannot be empty."
            )

        if not self.content.strip():
            raise ContextBuilderInputError(
                "content cannot be empty."
            )

        if self.rank <= 0:
            raise ContextBuilderInputError(
                "rank must be greater than zero."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context source."""
        return {
            "source_index": self.source_index,
            "chunk_id": self.chunk_id,
            "source_file_id": self.source_file_id,
            "repository_id": self.repository_id,
            "relative_path": self.relative_path,
            "language": self.language,
            "trust": self.trust,
            "retrieval_score": self.retrieval_score,
            "rank": self.rank,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "is_documentation": self.is_documentation,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BuiltContext:
    """Represents formatted LLM context and its provenance."""

    text: str
    sources: tuple[ContextSource, ...]
    query: str | None = None
    total_characters: int = 0
    total_lines: int = 0
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_count(self) -> int:
        """Return the number of context sources."""
        return len(self.sources)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return chunk IDs included in the context."""
        return tuple(
            source.chunk_id
            for source in self.sources
        )

    @property
    def repository_ids(self) -> tuple[str, ...]:
        """Return unique repository IDs in source order."""
        return tuple(
            dict.fromkeys(
                source.repository_id
                for source in self.sources
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the built context."""
        return {
            "text": self.text,
            "sources": [
                source.to_dict()
                for source in self.sources
            ],
            "query": self.query,
            "source_count": self.source_count,
            "chunk_ids": list(self.chunk_ids),
            "repository_ids": list(self.repository_ids),
            "total_characters": self.total_characters,
            "total_lines": self.total_lines,
            "truncated": self.truncated,
            "metadata": dict(self.metadata),
        }


class ContextBuilder:
    """Build structured, provenance-preserving LLM context."""

    DEFAULT_SEPARATOR = "\n\n"
    DEFAULT_SOURCE_HEADER = "[Source {index}]"

    def __init__(
        self,
        max_characters: int | None = None,
        separator: str = DEFAULT_SEPARATOR,
        include_metadata: bool = True,
        include_scores: bool = True,
        include_trust: bool = True,
    ) -> None:
        if max_characters is not None:
            if (
                not isinstance(max_characters, int)
                or isinstance(max_characters, bool)
            ):
                raise ContextBuilderInputError(
                    "max_characters must be an integer or None."
                )

            if max_characters <= 0:
                raise ContextBuilderInputError(
                    "max_characters must be greater than zero."
                )

        if not isinstance(separator, str):
            raise ContextBuilderInputError(
                "separator must be a string."
            )

        self._max_characters = max_characters
        self._separator = separator
        self._include_metadata = include_metadata
        self._include_scores = include_scores
        self._include_trust = include_trust

    @property
    def max_characters(self) -> int | None:
        """Return the maximum context size."""
        return self._max_characters

    @property
    def separator(self) -> str:
        """Return the source separator."""
        return self._separator

    def build(
        self,
        results: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
        query: str | None = None,
    ) -> BuiltContext:
        """Build LLM context from retrieved chunks."""
        self._validate_results(results)

        if query is not None:
            if not isinstance(query, str):
                raise ContextBuilderInputError(
                    "query must be a string or None."
                )

            query = query.strip()

            if not query:
                raise ContextBuilderInputError(
                    "query cannot be empty when provided."
                )

        sources: list[ContextSource] = []

        for source_index, result in enumerate(
            results,
            start=1,
        ):
            sources.append(
                self._to_context_source(
                    result,
                    source_index,
                )
            )

        selected_sources, truncated = (
            self._select_sources(sources)
        )

        text = self._format_sources(
            selected_sources
        )

        return BuiltContext(
            text=text,
            sources=tuple(selected_sources),
            query=query,
            total_characters=len(text),
            total_lines=self._count_lines(text),
            truncated=truncated,
            metadata={
                "max_characters": self._max_characters,
                "include_metadata": self._include_metadata,
                "include_scores": self._include_scores,
                "include_trust": self._include_trust,
                "separator_length": len(self._separator),
            },
        )

    def build_from_response(
        self,
        response: Any,
    ) -> BuiltContext:
        """Build context from a RetrievalResponse-like object."""
        if not hasattr(response, "results"):
            raise ContextBuilderInputError(
                "response must contain a results attribute."
            )

        results = response.results
        query = getattr(response, "query", None)

        return self.build(
            results=results,
            query=query,
        )

    def _validate_results(
        self,
        results: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
    ) -> None:
        """Validate retrieval results."""
        if not isinstance(results, (list, tuple)):
            raise ContextBuilderInputError(
                "results must be a list or tuple."
            )

        for result in results:
            if not isinstance(result, RetrievedChunk):
                raise ContextBuilderInputError(
                    "results must contain only RetrievedChunk objects."
                )

    def _to_context_source(
        self,
        result: RetrievedChunk,
        source_index: int,
    ) -> ContextSource:
        """Convert a RetrievedChunk into a provenance record."""
        chunk = result.chunk

        relative_path = str(
            chunk.metadata.get(
                "relative_path",
                chunk.source_file_id,
            )
        )

        metadata = dict(
            chunk.metadata
        )

        metadata.update(
            {
                "retriever_name": result.retriever_name,
                "rerank_score": result.rerank_score,
            }
        )

        return ContextSource(
            source_index=source_index,
            chunk_id=chunk.chunk_id,
            source_file_id=chunk.source_file_id,
            repository_id=chunk.repository_id,
            relative_path=relative_path,
            language=chunk.language.value,
            trust=chunk.trust.value,
            retrieval_score=result.retrieval_score,
            rank=result.rank,
            symbol_name=chunk.symbol_name,
            symbol_type=chunk.symbol_type,
            is_documentation=chunk.is_documentation,
            content=chunk.content,
            metadata=metadata,
        )

    def _select_sources(
        self,
        sources: list[ContextSource],
    ) -> tuple[list[ContextSource], bool]:
        """Select sources while respecting the character limit."""
        if self._max_characters is None:
            return sources, False

        selected: list[ContextSource] = []
        truncated = False
        current_length = 0

        for source in sources:
            rendered = self._format_source(
                source
            )

            additional_length = len(rendered)

            if selected:
                additional_length += len(
                    self._separator
                )

            if (
                current_length + additional_length
                <= self._max_characters
            ):
                selected.append(source)
                current_length += additional_length
                continue

            truncated = True
            break

        return selected, truncated

    def _format_sources(
        self,
        sources: list[ContextSource],
    ) -> str:
        """Format all selected sources."""
        return self._separator.join(
            self._format_source(source)
            for source in sources
        )

    def _format_source(
        self,
        source: ContextSource,
    ) -> str:
        """Format one context source."""
        lines = [
            self.DEFAULT_SOURCE_HEADER.format(
                index=source.source_index
            )
        ]

        if self._include_metadata:
            lines.append(
                f"Path: {source.relative_path}"
            )
            lines.append(
                f"Language: {source.language}"
            )

        if self._include_trust:
            lines.append(
                f"Trust: {source.trust}"
            )

        if self._include_scores:
            lines.append(
                f"Retrieval Score: "
                f"{source.retrieval_score:.6f}"
            )

            if source.rank > 0:
                lines.append(
                    f"Rank: {source.rank}"
                )

        if source.symbol_name:
            lines.append(
                f"Symbol: {source.symbol_name}"
            )

        if source.symbol_type:
            lines.append(
                f"Symbol Type: {source.symbol_type}"
            )

        lines.extend(
            [
                "",
                source.content,
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _count_lines(text: str) -> int:
        """Count lines in formatted context."""
        if not text:
            return 0

        return len(
            text.splitlines()
        )

    def metadata(self) -> dict[str, Any]:
        """Return context-builder configuration."""
        return {
            "max_characters": self._max_characters,
            "separator": self._separator,
            "include_metadata": self._include_metadata,
            "include_scores": self._include_scores,
            "include_trust": self._include_trust,
        }

    def __repr__(self) -> str:
        """Return a useful developer representation."""
        return (
            "ContextBuilder("
            f"max_characters={self._max_characters!r}, "
            f"include_metadata={self._include_metadata}, "
            f"include_scores={self._include_scores}, "
            f"include_trust={self._include_trust}"
            ")"
        )