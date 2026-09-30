"""
Metadata extraction utilities for SecureCodeRAG.

This module converts eligible repository files into SourceFile objects.
It records deterministic file identity, repository-relative paths,
content hashes, language information, documentation status, and
research trust metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.language_detector import (
    LanguageDetectionResult,
    LanguageDetector,
)
from src.models import (
    ProgrammingLanguage,
    SourceFile,
    SourceTrust,
)


class MetadataExtractionError(Exception):
    """Base exception for metadata extraction failures."""


class FileReadError(MetadataExtractionError):
    """Raised when a source file cannot be read."""


@dataclass(frozen=True)
class MetadataConfig:
    """
    Configuration for source-file metadata extraction.

    Attributes
    ----------
    encoding:
        Primary encoding used to read source files.

    include_content:
        Whether the complete file content should be stored in SourceFile.

    trust:
        Default trust level assigned to newly ingested files.
    """

    encoding: str = "utf-8"
    include_content: bool = True
    trust: SourceTrust = SourceTrust.TRUSTED


class MetadataExtractor:
    """
    Extract deterministic metadata and create SourceFile objects.
    """

    DOCUMENTATION_EXTENSIONS = frozenset(
        {
            ".md",
            ".markdown",
            ".rst",
            ".adoc",
            ".txt",
            ".text",
        }
    )

    DOCUMENTATION_FILENAMES = frozenset(
        {
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
    )

    CONFIGURATION_EXTENSIONS = frozenset(
        {
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".xml",
        }
    )

    def __init__(
        self,
        config: MetadataConfig | None = None,
        language_detector: LanguageDetector | None = None,
    ) -> None:
        """
        Initialize the metadata extractor.

        Parameters
        ----------
        config:
            Optional metadata configuration.

        language_detector:
            Optional language detector instance.
        """
        self.config = config or MetadataConfig()
        self.language_detector = (
            language_detector or LanguageDetector()
        )

    def extract(
        self,
        repository_id: str,
        repository_root: str | Path,
        file_path: str | Path,
    ) -> SourceFile:
        """
        Extract metadata and construct a SourceFile.

        Parameters
        ----------
        repository_id:
            Stable repository identifier.

        repository_root:
            Absolute repository root.

        file_path:
            File to process.

        Returns
        -------
        SourceFile
            Fully populated source-file model.

        Raises
        ------
        MetadataExtractionError
            If the file cannot be processed safely.
        """
        if not repository_id.strip():
            raise MetadataExtractionError(
                "repository_id cannot be empty."
            )

        root = self._resolve_root(repository_root)
        path = self._resolve_file(file_path)

        relative_path = self._relative_path(
            root,
            path,
        )

        content = self._read_content(path)

        try:
            size_bytes = path.stat().st_size
        except OSError as exc:
            raise MetadataExtractionError(
                f"Could not read file metadata: {path}"
            ) from exc

        sha256 = self._calculate_sha256(content)

        language_result = self.language_detector.detect(path)

        file_id = self._build_file_id(
            repository_id=repository_id,
            relative_path=relative_path,
            sha256=sha256,
        )

        is_documentation = self._is_documentation(
            path,
            language_result,
        )

        metadata = self._build_metadata(
            path=path,
            relative_path=relative_path,
            language_result=language_result,
            is_documentation=is_documentation,
        )

        stored_content = content if self.config.include_content else ""

        return SourceFile(
            file_id=file_id,
            repository_id=repository_id,
            relative_path=relative_path.as_posix(),
            language=language_result.language,
            content=stored_content,
            size_bytes=size_bytes,
            sha256=sha256,
            trust=self.config.trust,
            is_documentation=is_documentation,
            metadata=metadata,
        )

    def extract_all(
        self,
        repository_id: str,
        repository_root: str | Path,
        file_paths: list[Path] | tuple[Path, ...],
    ) -> tuple[SourceFile, ...]:
        """
        Extract metadata for multiple files.

        Results are deterministic and sorted by repository-relative path.
        """
        source_files = [
            self.extract(
                repository_id=repository_id,
                repository_root=repository_root,
                file_path=file_path,
            )
            for file_path in file_paths
        ]

        return tuple(
            sorted(
                source_files,
                key=lambda source: source.relative_path.lower(),
            )
        )

    @staticmethod
    def _resolve_root(
        repository_root: str | Path,
    ) -> Path:
        """Resolve and validate the repository root."""
        root = Path(repository_root).expanduser()

        if not root.exists():
            raise MetadataExtractionError(
                f"Repository root does not exist: {root}"
            )

        if not root.is_dir():
            raise MetadataExtractionError(
                f"Repository root is not a directory: {root}"
            )

        try:
            return root.resolve(strict=True)
        except OSError as exc:
            raise MetadataExtractionError(
                f"Could not resolve repository root: {root}"
            ) from exc

    @staticmethod
    def _resolve_file(
        file_path: str | Path,
    ) -> Path:
        """Resolve and validate a source file."""
        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileReadError(
                f"Source file does not exist: {path}"
            )

        if not path.is_file():
            raise FileReadError(
                f"Source path is not a regular file: {path}"
            )

        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise FileReadError(
                f"Could not resolve source file: {path}"
            ) from exc

    @staticmethod
    def _relative_path(
        repository_root: Path,
        file_path: Path,
    ) -> Path:
        """Ensure the file belongs to the repository."""
        try:
            return file_path.relative_to(repository_root)
        except ValueError as exc:
            raise MetadataExtractionError(
                "Source file is outside repository boundary: "
                f"{file_path}"
            ) from exc

    def _read_content(
        self,
        file_path: Path,
    ) -> str:
        """Read a source file using the configured encoding."""
        try:
            return file_path.read_text(
                encoding=self.config.encoding,
                errors="strict",
            )
        except (OSError, UnicodeDecodeError) as exc:
            raise FileReadError(
                f"Could not decode source file as "
                f"{self.config.encoding}: {file_path}"
            ) from exc

    @staticmethod
    def _calculate_sha256(content: str) -> str:
        """Calculate a deterministic SHA-256 content hash."""
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _build_file_id(
        repository_id: str,
        relative_path: Path,
        sha256: str,
    ) -> str:
        """
        Build a deterministic file identifier.

        Both repository identity, relative path, and content hash
        participate in the identifier.
        """
        identity = (
            f"{repository_id}:"
            f"{relative_path.as_posix()}:"
            f"{sha256}"
        )

        digest = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        return f"file-{digest[:20]}"

    def _is_documentation(
        self,
        file_path: Path,
        language_result: LanguageDetectionResult,
    ) -> bool:
        """Determine whether a file should be treated as documentation."""
        filename = file_path.name.lower()
        extension = file_path.suffix.lower()

        if filename in self.DOCUMENTATION_FILENAMES:
            return True

        if extension in self.DOCUMENTATION_EXTENSIONS:
            return True

        return language_result.language is ProgrammingLanguage.UNKNOWN and filename.startswith(
            (
                "readme",
                "changelog",
                "contributing",
            )
        )

    def _build_metadata(
        self,
        path: Path,
        relative_path: Path,
        language_result: LanguageDetectionResult,
        is_documentation: bool,
    ) -> dict[str, object]:
        """Build structured metadata for downstream pipeline stages."""
        extension = path.suffix.lower()

        if is_documentation:
            file_type = "documentation"
        elif extension in self.CONFIGURATION_EXTENSIONS:
            file_type = "configuration"
        elif language_result.language is ProgrammingLanguage.UNKNOWN:
            file_type = "unknown"
        else:
            file_type = "source_code"

        return {
            "filename": path.name,
            "extension": extension,
            "relative_directory": (
                relative_path.parent.as_posix()
                if relative_path.parent != Path(".")
                else ""
            ),
            "file_type": file_type,
            "language_detection_method": (
                language_result.method
            ),
            "language_detection_confidence": (
                language_result.confidence
            ),
        }