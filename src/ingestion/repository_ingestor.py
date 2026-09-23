"""
Repository ingestion orchestration for SecureCodeRAG.

This module combines the existing ingestion primitives:

    RepositoryLoader
        -> FileFilter
        -> LanguageDetector
        -> SourceFile
        -> RepositorySnapshot

It is intentionally responsible only for deterministic local repository
ingestion. Parsing, chunking, embeddings, retrieval, poisoning, and
generation belong to later pipeline stages.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.file_filter import (
    FileFilter,
    FileFilterConfig,
)
from src.ingestion.language_detector import (
    LanguageDetector,
)
from src.ingestion.models import (
    RepositorySnapshot,
    SourceFile,
)
from src.ingestion.repository_loader import (
    RepositoryInfo,
    RepositoryLoader,
)


class RepositoryIngestionError(Exception):
    """Base exception for repository ingestion failures."""


class SourceFileReadError(RepositoryIngestionError):
    """Raised when an eligible source file cannot be read."""


@dataclass(frozen=True)
class RepositoryIngestionConfig:
    """
    Configuration for repository ingestion.

    Attributes
    ----------
    file_filter:
        Configuration used to determine eligible files.

    encoding:
        Encoding used when reading source files.

    include_content:
        Whether complete source content should be retained.

    fail_fast:
        Whether one unreadable file should stop the complete ingestion.
    """

    file_filter: FileFilterConfig = field(
        default_factory=FileFilterConfig
    )
    encoding: str = "utf-8"
    include_content: bool = True
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if not self.encoding.strip():
            raise ValueError("encoding cannot be empty.")


class RepositoryIngestor:
    """
    Deterministic local repository ingestion orchestrator.

    The ingestor does not:

    - modify repository files
    - access the network
    - parse ASTs
    - create chunks
    - generate embeddings
    - perform retrieval
    - execute generated code
    - apply poisoning

    Those responsibilities belong to later pipeline components.
    """

    def __init__(
        self,
        config: RepositoryIngestionConfig | None = None,
        repository_loader: RepositoryLoader | None = None,
        file_filter: FileFilter | None = None,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        """
        Initialize the repository ingestor.

        Parameters
        ----------
        config:
            Optional ingestion configuration.

        repository_loader:
            Optional repository loader.

        file_filter:
            Optional file filter.

        language_detector:
            Optional language detector.
        """
        self.config = config or RepositoryIngestionConfig()

        self.repository_loader = (
            repository_loader or RepositoryLoader()
        )

        self.file_filter = (
            file_filter
            or FileFilter(self.config.file_filter)
        )

        self.language_detector = (
            language_detector or LanguageDetector()
        )

    def ingest(
        self,
        repository_path: str | Path,
    ) -> RepositorySnapshot:
        """
        Ingest an authorized local repository.

        The operation is deterministic for a fixed repository state and
        configuration.

        Parameters
        ----------
        repository_path:
            Local repository directory.

        Returns
        -------
        RepositorySnapshot
            Snapshot containing successfully ingested files and
            ingestion statistics.
        """
        repository_info = self.repository_loader.load(
            repository_path
        )

        evaluations = self.file_filter.evaluate_all(
            repository_info.file_paths
        )

        accepted_paths = tuple(
            result.path
            for result in evaluations
            if result.accepted
        )

        skipped_files = len(
            repository_info.file_paths
        ) - len(accepted_paths)

        source_files: list[SourceFile] = []
        failed_files = 0

        for file_path in accepted_paths:
            try:
                source_file = self._build_source_file(
                    repository_info=repository_info,
                    file_path=file_path,
                )
            except SourceFileReadError:
                failed_files += 1

                if self.config.fail_fast:
                    raise

                continue

            source_files.append(source_file)

        source_files.sort(
            key=lambda source: source.relative_path.lower()
        )

        metadata = {
            "filter_accepted_files": len(accepted_paths),
            "filter_skipped_files": skipped_files,
            "ingestion_failed_files": failed_files,
            "encoding": self.config.encoding,
            "include_content": self.config.include_content,
        }

        return RepositorySnapshot(
            repository_id=repository_info.repository_id,
            root_path=repository_info.root_path.as_posix(),
            files=tuple(source_files),
            total_files=repository_info.file_count,
            skipped_files=skipped_files,
            failed_files=failed_files,
            metadata=metadata,
        )

    def _build_source_file(
        self,
        repository_info: RepositoryInfo,
        file_path: Path,
    ) -> SourceFile:
        """
        Convert one accepted repository file into SourceFile.
        """
        try:
            content = file_path.read_text(
                encoding=self.config.encoding,
                errors="strict",
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise SourceFileReadError(
                f"Could not read source file: {file_path}"
            ) from exc

        relative_path = file_path.relative_to(
            repository_info.root_path
        )

        language_result = self.language_detector.detect(
            file_path,
            content,
        )

        content_hash = self._calculate_content_hash(
            content
        )

        file_id = self._build_file_id(
            repository_id=repository_info.repository_id,
            relative_path=relative_path,
            content_hash=content_hash,
        )

        try:
            size_bytes = file_path.stat().st_size
        except OSError as exc:
            raise SourceFileReadError(
                f"Could not read file metadata: {file_path}"
            ) from exc

        line_count = self._calculate_line_count(
            content
        )

        stored_content = (
            content
            if self.config.include_content
            else ""
        )

        metadata = {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "language_detection_method": (
                language_result.method
            ),
            "language_detection_confidence": (
                language_result.confidence
            ),
            "is_documentation": (
                self._is_documentation(file_path)
            ),
        }

        return SourceFile(
            file_id=file_id,
            repository_id=repository_info.repository_id,
            relative_path=relative_path.as_posix(),
            absolute_path=file_path.as_posix(),
            language=language_result.language.value,
            content=stored_content,
            content_hash=content_hash,
            size_bytes=size_bytes,
            line_count=line_count,
            is_binary=False,
            metadata=metadata,
        )

    @staticmethod
    def _calculate_content_hash(
        content: str,
    ) -> str:
        """Calculate a deterministic SHA-256 content hash."""
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _build_file_id(
        repository_id: str,
        relative_path: Path,
        content_hash: str,
    ) -> str:
        """
        Build a deterministic identifier for a source file.
        """
        identity = (
            f"{repository_id}:"
            f"{relative_path.as_posix()}:"
            f"{content_hash}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        return f"file-{digest[:20]}"

    @staticmethod
    def _calculate_line_count(
        content: str,
    ) -> int:
        """
        Calculate the number of logical source lines.

        Empty content has zero lines. A final newline does not create
        an additional logical source line.
        """
        if not content:
            return 0

        return len(content.splitlines())

    @staticmethod
    def _is_documentation(
        file_path: Path,
    ) -> bool:
        """Determine whether the file is documentation."""
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()

        documentation_extensions = {
            ".md",
            ".markdown",
            ".rst",
            ".adoc",
            ".txt",
            ".text",
        }

        documentation_names = {
            "readme",
            "readme.md",
            "readme.txt",
            "readme.rst",
            "changelog",
            "changelog.md",
            "contributing",
            "contributing.md",
            "license",
            "license.md",
        }

        return (
            filename in documentation_names
            or extension in documentation_extensions
        )