"""
Repository loading utilities for SecureCodeRAG.

This module is responsible for discovering files inside an authorized
local repository. It deliberately does not perform parsing, embedding,
retrieval, generation, or poisoning.

The loader is deterministic and keeps repository boundaries explicit so
that clean and poisoned research datasets can remain isolated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class RepositoryLoaderError(Exception):
    """Base exception for repository loading failures."""


class RepositoryNotFoundError(RepositoryLoaderError):
    """Raised when the repository path does not exist."""


class InvalidRepositoryError(RepositoryLoaderError):
    """Raised when the supplied repository path is not a directory."""


@dataclass(frozen=True)
class RepositoryInfo:
    """
    Metadata describing a loaded repository.

    Attributes
    ----------
    repository_id:
        Stable identifier derived from the normalized repository path.

    root_path:
        Absolute normalized repository root.

    file_paths:
        Deterministically ordered candidate files.

    file_count:
        Number of candidate files discovered.
    """

    repository_id: str
    root_path: Path
    file_paths: tuple[Path, ...]
    file_count: int


class RepositoryLoader:
    """
    Discover files from an authorized local repository.

    The loader does not perform network access and does not modify the
    repository.
    """

    DEFAULT_EXCLUDED_DIRECTORIES = frozenset(
        {
            ".git",
            ".hg",
            ".svn",
            ".idea",
            ".vscode",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "dist",
            "build",
            "target",
            ".next",
            "coverage",
            ".coverage",
        }
    )

    def __init__(
        self,
        excluded_directories: set[str] | frozenset[str] | None = None,
    ) -> None:
        """
        Initialize the repository loader.

        Parameters
        ----------
        excluded_directories:
            Optional directory names that should not be traversed.
            When omitted, the SecureCodeRAG defaults are used.
        """
        if excluded_directories is None:
            self._excluded_directories = (
                self.DEFAULT_EXCLUDED_DIRECTORIES
            )
        else:
            self._excluded_directories = frozenset(
                directory.strip()
                for directory in excluded_directories
                if directory.strip()
            )

    @staticmethod
    def normalize_root(repository_path: str | Path) -> Path:
        """
        Normalize and validate the repository root path.

        Parameters
        ----------
        repository_path:
            Local repository directory.

        Returns
        -------
        Path
            Absolute resolved repository path.

        Raises
        ------
        RepositoryNotFoundError
            If the path does not exist.

        InvalidRepositoryError
            If the path exists but is not a directory.
        """
        root = Path(repository_path).expanduser()

        if not root.exists():
            raise RepositoryNotFoundError(
                f"Repository does not exist: {root}"
            )

        if not root.is_dir():
            raise InvalidRepositoryError(
                f"Repository path is not a directory: {root}"
            )

        try:
            return root.resolve(strict=True)
        except OSError as exc:
            raise RepositoryLoaderError(
                f"Could not resolve repository path: {root}"
            ) from exc

    @staticmethod
    def build_repository_id(root_path: Path) -> str:
        """
        Build a stable repository identifier.

        The identifier is derived from the normalized absolute path.
        Repository contents are intentionally not hashed here because
        content hashing belongs to source-file metadata generation.
        """
        normalized = str(root_path).replace("\\", "/").lower()
        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        return f"repo-{digest[:16]}"

    def discover_files(
        self,
        repository_path: str | Path,
    ) -> tuple[Path, ...]:
        """
        Recursively discover candidate files.

        Files are returned in deterministic lexicographic order.

        Parameters
        ----------
        repository_path:
            Local repository directory.

        Returns
        -------
        tuple[Path, ...]
            Absolute paths of discovered files.
        """
        root = self.normalize_root(repository_path)

        discovered: list[Path] = []

        for current_root, directories, filenames in __import__(
            "os"
        ).walk(root):
            current_path = Path(current_root)

            directories[:] = sorted(
                directory
                for directory in directories
                if directory not in self._excluded_directories
            )

            for filename in sorted(filenames):
                file_path = current_path / filename

                if not file_path.is_file():
                    continue

                discovered.append(file_path.resolve())

        discovered.sort(
            key=lambda path: path.relative_to(root)
            .as_posix()
            .lower()
        )

        return tuple(discovered)

    def load(
        self,
        repository_path: str | Path,
    ) -> RepositoryInfo:
        """
        Load repository metadata and discover candidate files.

        Parameters
        ----------
        repository_path:
            Local repository directory.

        Returns
        -------
        RepositoryInfo
            Repository identity and discovered files.
        """
        root = self.normalize_root(repository_path)

        repository_id = self.build_repository_id(root)

        file_paths = self.discover_files(root)

        return RepositoryInfo(
            repository_id=repository_id,
            root_path=root,
            file_paths=file_paths,
            file_count=len(file_paths),
        )

    @staticmethod
    def relative_path(
        repository_root: str | Path,
        file_path: str | Path,
    ) -> Path:
        """
        Return a file path relative to the repository root.

        This method also verifies that the file belongs to the
        repository boundary.

        Parameters
        ----------
        repository_root:
            Repository root.

        file_path:
            File whose relative path is required.

        Returns
        -------
        Path
            Repository-relative file path.

        Raises
        ------
        RepositoryLoaderError
            If the file is outside the repository root.
        """
        root = RepositoryLoader.normalize_root(repository_root)
        candidate = Path(file_path).expanduser()

        try:
            resolved_candidate = candidate.resolve(strict=True)
        except OSError as exc:
            raise RepositoryLoaderError(
                f"Could not resolve file path: {candidate}"
            ) from exc

        try:
            return resolved_candidate.relative_to(root)
        except ValueError as exc:
            raise RepositoryLoaderError(
                "File is outside the repository boundary: "
                f"{resolved_candidate}"
            ) from exc