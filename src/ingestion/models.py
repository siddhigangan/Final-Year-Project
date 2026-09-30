"""Data models for repository ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    """Represents one source file discovered during repository ingestion."""

    file_id: str
    repository_id: str
    relative_path: str
    absolute_path: str
    language: str
    content: str
    content_hash: str
    size_bytes: int
    line_count: int
    is_binary: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def file_name(self) -> str:
        """Return the file name without its parent directories."""
        return Path(self.relative_path).name

    @property
    def extension(self) -> str:
        """Return the lowercase file extension."""
        return Path(self.relative_path).suffix.lower()

    @property
    def is_empty(self) -> bool:
        """Return whether the source file contains no content."""
        return not self.content

    def to_dict(self) -> dict[str, object]:
        """Serialize the source file into a JSON-compatible dictionary."""
        return {
            "file_id": self.file_id,
            "repository_id": self.repository_id,
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "language": self.language,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "is_binary": self.is_binary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RepositorySnapshot:
    """Represents the result of ingesting a repository."""

    repository_id: str
    root_path: str
    files: tuple[SourceFile, ...]
    total_files: int
    skipped_files: int
    failed_files: int
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def successful_files(self) -> int:
        """Return the number of successfully ingested files."""
        return len(self.files)

    @property
    def total_lines(self) -> int:
        """Return the total number of source lines."""
        return sum(source_file.line_count for source_file in self.files)

    def to_dict(self) -> dict[str, object]:
        """Serialize the repository snapshot."""
        return {
            "repository_id": self.repository_id,
            "root_path": self.root_path,
            "files": [
                source_file.to_dict()
                for source_file in self.files
            ],
            "total_files": self.total_files,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
            "successful_files": self.successful_files,
            "total_lines": self.total_lines,
            "metadata": dict(self.metadata),
        }