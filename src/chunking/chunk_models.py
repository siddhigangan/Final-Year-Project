"""Data models for AST-aware code chunking in SecureCodeRAG."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from src.models import ProgrammingLanguage, SourceTrust


class ChunkModelError(ValueError):
    """Raised when a chunk model contains invalid data."""


@dataclass(frozen=True)
class ASTNodeInfo:
    """Serializable structural information about an AST node."""

    node_type: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    is_named: bool = True

    def __post_init__(self) -> None:
        """Validate AST node information."""
        if not self.node_type.strip():
            raise ChunkModelError("node_type must not be empty")

        if self.start_byte < 0:
            raise ChunkModelError("start_byte must be non-negative")

        if self.end_byte < self.start_byte:
            raise ChunkModelError(
                "end_byte must be greater than or equal to start_byte"
            )

        if self.start_line < 1:
            raise ChunkModelError("start_line must be at least 1")

        if self.end_line < self.start_line:
            raise ChunkModelError(
                "end_line must be greater than or equal to start_line"
            )

        if self.start_column < 0:
            raise ChunkModelError("start_column must be non-negative")

        if self.end_column < 0:
            raise ChunkModelError("end_column must be non-negative")

    @property
    def byte_length(self) -> int:
        """Return the number of bytes covered by the node."""
        return self.end_byte - self.start_byte

    @property
    def line_count(self) -> int:
        """Return the number of source lines covered by the node."""
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert the AST node information to a dictionary."""
        return {
            "node_type": self.node_type,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "is_named": self.is_named,
            "byte_length": self.byte_length,
            "line_count": self.line_count,
        }


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata used during chunking and downstream retrieval."""

    repository_id: str
    source_file_id: str
    relative_path: str
    language: ProgrammingLanguage
    trust: SourceTrust = SourceTrust.UNKNOWN
    is_documentation: bool = False
    symbol_name: str | None = None
    symbol_type: str | None = None
    parent_symbol: str | None = None
    ast_node_type: str | None = None
    chunk_index: int = 0
    total_chunks: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk metadata."""
        if not self.repository_id.strip():
            raise ChunkModelError("repository_id must not be empty")

        if not self.source_file_id.strip():
            raise ChunkModelError("source_file_id must not be empty")

        if not self.relative_path.strip():
            raise ChunkModelError("relative_path must not be empty")

        if not isinstance(self.language, ProgrammingLanguage):
            raise ChunkModelError(
                "language must be a ProgrammingLanguage value"
            )

        if not isinstance(self.trust, SourceTrust):
            raise ChunkModelError("trust must be a SourceTrust value")

        if self.chunk_index < 0:
            raise ChunkModelError("chunk_index must be non-negative")

        if self.total_chunks is not None and self.total_chunks < 1:
            raise ChunkModelError("total_chunks must be at least 1")

        if (
            self.total_chunks is not None
            and self.chunk_index >= self.total_chunks
        ):
            raise ChunkModelError(
                "chunk_index must be smaller than total_chunks"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return {
            "repository_id": self.repository_id,
            "source_file_id": self.source_file_id,
            "relative_path": self.relative_path,
            "language": self.language.value,
            "trust": self.trust.value,
            "is_documentation": self.is_documentation,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "parent_symbol": self.parent_symbol,
            "ast_node_type": self.ast_node_type,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class ASTCodeChunk:
    """A code chunk generated from an AST node."""

    chunk_id: str
    content: str
    ast: ASTNodeInfo
    metadata: ChunkMetadata

    def __post_init__(self) -> None:
        """Validate the AST-aware chunk."""
        if not self.chunk_id.strip():
            raise ChunkModelError("chunk_id must not be empty")

        if not self.content.strip():
            raise ChunkModelError("content must not be empty")

        if not isinstance(self.ast, ASTNodeInfo):
            raise ChunkModelError("ast must be an ASTNodeInfo value")

        if not isinstance(self.metadata, ChunkMetadata):
            raise ChunkModelError(
                "metadata must be a ChunkMetadata value"
            )

    @property
    def language(self) -> ProgrammingLanguage:
        """Return the programming language of the chunk."""
        return self.metadata.language

    @property
    def start_line(self) -> int:
        """Return the first source line covered by the chunk."""
        return self.ast.start_line

    @property
    def end_line(self) -> int:
        """Return the last source line covered by the chunk."""
        return self.ast.end_line

    @property
    def line_count(self) -> int:
        """Return the number of source lines covered by the chunk."""
        return self.ast.line_count

    @property
    def symbol_name(self) -> str | None:
        """Return the symbol associated with the chunk."""
        return self.metadata.symbol_name

    @property
    def trust(self) -> SourceTrust:
        """Return the trust classification of the chunk."""
        return self.metadata.trust

    def to_code_chunk(self) -> Any:
        """Convert this model to the system-level CodeChunk model."""
        from src.models import CodeChunk

        return CodeChunk(
            chunk_id=self.chunk_id,
            source_file_id=self.metadata.source_file_id,
            repository_id=self.metadata.repository_id,
            content=self.content,
            language=self.metadata.language,
            start_line=self.ast.start_line,
            end_line=self.ast.end_line,
            symbol_name=self.metadata.symbol_name,
            symbol_type=self.metadata.symbol_type,
            ast_node_type=self.metadata.ast_node_type
            or self.ast.node_type,
            parent_symbol=self.metadata.parent_symbol,
            is_documentation=self.metadata.is_documentation,
            trust=self.metadata.trust,
            metadata={
                **self.metadata.extra,
                "relative_path": self.metadata.relative_path,
                "ast": self.ast.to_dict(),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the AST-aware chunk to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "ast": self.ast.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


def build_chunk_id(
    source_file_id: str,
    start_byte: int,
    end_byte: int,
    content: str,
) -> str:
    """Build a deterministic identifier for a code chunk."""
    if not source_file_id.strip():
        raise ChunkModelError("source_file_id must not be empty")

    if start_byte < 0:
        raise ChunkModelError("start_byte must be non-negative")

    if end_byte < start_byte:
        raise ChunkModelError(
            "end_byte must be greater than or equal to start_byte"
        )

    if not content:
        raise ChunkModelError("content must not be empty")

    payload = (
        f"{source_file_id}:{start_byte}:{end_byte}:"
        f"{content}"
    )

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return f"chunk-{digest[:20]}"


__all__ = [
    "ASTCodeChunk",
    "ASTNodeInfo",
    "ChunkMetadata",
    "ChunkModelError",
    "build_chunk_id",
]