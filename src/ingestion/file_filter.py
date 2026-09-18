"""
File filtering utilities for SecureCodeRAG.

This module determines which discovered repository files are eligible
for ingestion into the research corpus.

Filtering is intentionally deterministic and conservative. The module
does not perform language detection, parsing, embedding, or indexing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileFilterConfig:
    """
    Configuration controlling repository file filtering.

    Attributes
    ----------
    max_file_size_bytes:
        Maximum permitted file size.

    allowed_extensions:
        File extensions considered suitable for text/code ingestion.

    allowed_filenames:
        Extensionless or special filenames that may be ingested.

    excluded_extensions:
        File extensions that are always rejected.

    excluded_filenames:
        Specific filenames that are rejected regardless of extension.

    reject_secret_patterns:
        Whether obvious secret-like filenames should be rejected.
    """

    max_file_size_bytes: int = 1_000_000

    allowed_extensions: frozenset[str] = frozenset(
        {
            ".py",
            ".pyw",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",
            ".ts",
            ".tsx",
            ".java",
            ".c",
            ".h",
            ".cc",
            ".cpp",
            ".cxx",
            ".hpp",
            ".hh",
            ".hxx",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".kts",
            ".scala",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".sql",
            ".r",
            ".R",
            ".dart",
            ".lua",
            ".pl",
            ".pm",
            ".ex",
            ".exs",
            ".erl",
            ".hrl",
            ".clj",
            ".cljs",
            ".groovy",
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".xml",
            ".json",
            ".jsonl",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".md",
            ".markdown",
            ".rst",
            ".txt",
            ".adoc",
            ".text",
            ".csv",
        }
    )

    allowed_filenames: frozenset[str] = frozenset(
        {
            "Dockerfile",
            "Makefile",
            "CMakeLists.txt",
            "Jenkinsfile",
            "Procfile",
            "Gemfile",
            "Rakefile",
            "Vagrantfile",
            "LICENSE",
            "LICENSE.md",
            "README",
            "README.md",
            "README.txt",
        }
    )

    excluded_extensions: frozenset[str] = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".webp",
            ".svg",
            ".mp3",
            ".wav",
            ".ogg",
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".7z",
            ".rar",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".jar",
            ".war",
            ".class",
            ".pyc",
            ".pyo",
            ".so",
            ".dll",
            ".dylib",
            ".exe",
            ".bin",
            ".db",
            ".sqlite",
            ".sqlite3",
        }
    )

    excluded_filenames: frozenset[str] = frozenset(
        {
            ".env",
            ".env.local",
            ".env.development",
            ".env.production",
            ".env.test",
            "id_rsa",
            "id_rsa.pub",
            "id_dsa",
            "id_ecdsa",
            "id_ed25519",
        }
    )

    reject_secret_patterns: bool = True

    def __post_init__(self) -> None:
        """Validate filter configuration."""
        if self.max_file_size_bytes <= 0:
            raise ValueError(
                "max_file_size_bytes must be greater than zero."
            )


@dataclass(frozen=True)
class FileFilterResult:
    """
    Result of evaluating a repository file.

    Attributes
    ----------
    path:
        Absolute file path.

    accepted:
        Whether the file is eligible for ingestion.

    reason:
        Human-readable reason for the filtering decision.
    """

    path: Path
    accepted: bool
    reason: str


class FileFilter:
    """
    Deterministic filter for repository files.

    The filter works only on local filesystem metadata and file names.
    """

    SECRET_PATTERNS = (
        re.compile(r"^\.env(?:\..+)?$", re.IGNORECASE),
        re.compile(r".*\.pem$", re.IGNORECASE),
        re.compile(r".*\.key$", re.IGNORECASE),
        re.compile(r".*\.p12$", re.IGNORECASE),
        re.compile(r".*\.pfx$", re.IGNORECASE),
        re.compile(r".*credentials.*", re.IGNORECASE),
        re.compile(r".*secret.*", re.IGNORECASE),
        re.compile(r".*password.*", re.IGNORECASE),
        re.compile(r".*passwd.*", re.IGNORECASE),
    )

    def __init__(
        self,
        config: FileFilterConfig | None = None,
    ) -> None:
        """
        Initialize the file filter.

        Parameters
        ----------
        config:
            Optional filtering configuration.
        """
        self.config = config or FileFilterConfig()

    def evaluate(
        self,
        file_path: str | Path,
    ) -> FileFilterResult:
        """
        Evaluate a single file.

        Parameters
        ----------
        file_path:
            Path to a repository file.

        Returns
        -------
        FileFilterResult
            Filtering decision and explanation.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="file does not exist",
            )

        if not path.is_file():
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="path is not a regular file",
            )

        filename = path.name
        extension = path.suffix.lower()

        if filename in self.config.excluded_filenames:
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="excluded filename",
            )

        if (
            self.config.reject_secret_patterns
            and self._looks_like_secret(filename)
        ):
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="possible secret or credential file",
            )

        try:
            size_bytes = path.stat().st_size
        except OSError:
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="unable to read file metadata",
            )

        if size_bytes > self.config.max_file_size_bytes:
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="file exceeds maximum allowed size",
            )

        if extension in self.config.excluded_extensions:
            return FileFilterResult(
                path=path,
                accepted=False,
                reason="excluded binary or generated extension",
            )

        if filename in self.config.allowed_filenames:
            return FileFilterResult(
                path=path,
                accepted=True,
                reason="allowed special filename",
            )

        if extension in self.config.allowed_extensions:
            return FileFilterResult(
                path=path,
                accepted=True,
                reason="allowed text or source extension",
            )

        if not extension and self._is_text_file(path):
            return FileFilterResult(
                path=path,
                accepted=True,
                reason="extensionless text file",
            )

        return FileFilterResult(
            path=path,
            accepted=False,
            reason="unsupported file extension",
        )

    def filter_files(
        self,
        file_paths: list[Path] | tuple[Path, ...],
    ) -> tuple[Path, ...]:
        """
        Filter a collection of repository files.

        Accepted paths are returned in deterministic order.

        Parameters
        ----------
        file_paths:
            Candidate repository files.

        Returns
        -------
        tuple[Path, ...]
            Accepted files.
        """
        accepted: list[Path] = []

        for file_path in file_paths:
            result = self.evaluate(file_path)

            if result.accepted:
                accepted.append(result.path.resolve())

        accepted.sort(
            key=lambda path: path.as_posix().lower()
        )

        return tuple(accepted)

    def evaluate_all(
        self,
        file_paths: list[Path] | tuple[Path, ...],
    ) -> tuple[FileFilterResult, ...]:
        """
        Evaluate all candidate files and preserve filtering reasons.

        Parameters
        ----------
        file_paths:
            Candidate repository files.

        Returns
        -------
        tuple[FileFilterResult, ...]
            Results in deterministic path order.
        """
        results = [
            self.evaluate(file_path)
            for file_path in file_paths
        ]

        results.sort(
            key=lambda result: result.path.as_posix().lower()
        )

        return tuple(results)

    def _looks_like_secret(self, filename: str) -> bool:
        """
        Determine whether a filename resembles a secret file.
        """
        return any(
            pattern.fullmatch(filename)
            for pattern in self.SECRET_PATTERNS
        )

    @staticmethod
    def _is_text_file(
        file_path: Path,
        sample_size: int = 4096,
    ) -> bool:
        """
        Perform a lightweight text/binary check.

        The method reads only a small prefix and does not load the entire
        file into memory.
        """
        try:
            with file_path.open("rb") as file:
                sample = file.read(sample_size)
        except OSError:
            return False

        if b"\x00" in sample:
            return False

        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return False

        return True