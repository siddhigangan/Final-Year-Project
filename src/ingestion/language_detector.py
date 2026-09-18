"""Programming-language detection for repository files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from src.models import ProgrammingLanguage


class LanguageDetectionError(ValueError):
    """Raised when language detection receives invalid input."""


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Result returned by the language detector."""

    path: Path
    language: ProgrammingLanguage
    confidence: float
    method: str


class LanguageDetector:
    """Detect supported programming languages from repository files."""

    EXTENSION_MAP: ClassVar[dict[str, ProgrammingLanguage]] = {
        ".py": ProgrammingLanguage.PYTHON,
        ".pyw": ProgrammingLanguage.PYTHON,
        ".js": ProgrammingLanguage.JAVASCRIPT,
        ".jsx": ProgrammingLanguage.JAVASCRIPT,
        ".mjs": ProgrammingLanguage.JAVASCRIPT,
        ".cjs": ProgrammingLanguage.JAVASCRIPT,
        ".ts": ProgrammingLanguage.TYPESCRIPT,
        ".tsx": ProgrammingLanguage.TYPESCRIPT,
        ".java": ProgrammingLanguage.JAVA,
        ".c": ProgrammingLanguage.C,
        ".h": ProgrammingLanguage.C,
        ".cc": ProgrammingLanguage.CPP,
        ".cpp": ProgrammingLanguage.CPP,
        ".cxx": ProgrammingLanguage.CPP,
        ".hpp": ProgrammingLanguage.CPP,
        ".hh": ProgrammingLanguage.CPP,
        ".hxx": ProgrammingLanguage.CPP,
        ".cs": ProgrammingLanguage.CSHARP,
        ".go": ProgrammingLanguage.GO,
        ".rs": ProgrammingLanguage.RUST,
        ".rb": ProgrammingLanguage.RUBY,
        ".php": ProgrammingLanguage.PHP,
    }

    SPECIAL_FILENAME_MAP: ClassVar[dict[str, ProgrammingLanguage]] = {
        "dockerfile": ProgrammingLanguage.UNKNOWN,
        "makefile": ProgrammingLanguage.UNKNOWN,
        "cmakelists.txt": ProgrammingLanguage.UNKNOWN,
        "procfile": ProgrammingLanguage.UNKNOWN,
        "license": ProgrammingLanguage.UNKNOWN,
        "license.md": ProgrammingLanguage.UNKNOWN,
        "readme": ProgrammingLanguage.UNKNOWN,
        "readme.md": ProgrammingLanguage.UNKNOWN,
        "readme.txt": ProgrammingLanguage.UNKNOWN,
    }

    SHEBANG_MAP: ClassVar[dict[str, ProgrammingLanguage]] = {
        "python": ProgrammingLanguage.PYTHON,
        "python3": ProgrammingLanguage.PYTHON,
        "python2": ProgrammingLanguage.PYTHON,
        "node": ProgrammingLanguage.JAVASCRIPT,
        "nodejs": ProgrammingLanguage.JAVASCRIPT,
        "ruby": ProgrammingLanguage.RUBY,
        "php": ProgrammingLanguage.PHP,
    }

    SHEBANG_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^#!\s*(?:/usr/bin/env\s+)?(?P<executable>[^\s]+)",
        re.IGNORECASE,
    )

    def detect(
        self,
        path: Path,
        content: str | None = None,
    ) -> LanguageDetectionResult:
        """Detect the programming language for a file."""
        if not isinstance(path, Path):
            path = Path(path)

        filename = path.name.lower()

        if filename in self.SPECIAL_FILENAME_MAP:
            return LanguageDetectionResult(
                path=path,
                language=self.SPECIAL_FILENAME_MAP[filename],
                confidence=1.0,
                method="filename",
            )

        extension = path.suffix.lower()

        if extension in self.EXTENSION_MAP:
            return LanguageDetectionResult(
                path=path,
                language=self.EXTENSION_MAP[extension],
                confidence=1.0,
                method="extension",
            )

        if content is not None:
            if not isinstance(content, str):
                raise LanguageDetectionError(
                    "content must be a string or None"
                )

            shebang_language = self._detect_from_shebang(content)

            if shebang_language is not None:
                return LanguageDetectionResult(
                    path=path,
                    language=shebang_language,
                    confidence=0.95,
                    method="shebang",
                )

        return LanguageDetectionResult(
            path=path,
            language=ProgrammingLanguage.UNKNOWN,
            confidence=0.0,
            method="unknown",
        )

    def detect_all(
        self,
        paths: list[Path],
        contents: dict[Path, str] | None = None,
    ) -> list[LanguageDetectionResult]:
        """Detect languages for multiple files deterministically."""
        if not isinstance(paths, list):
            raise LanguageDetectionError("paths must be a list")

        contents = contents or {}

        normalized_paths = [
            path if isinstance(path, Path) else Path(path)
            for path in paths
        ]

        results: list[LanguageDetectionResult] = []

        for path in sorted(
            normalized_paths,
            key=lambda item: item.as_posix().lower(),
        ):
            results.append(
                self.detect(
                    path,
                    contents.get(path),
                )
            )

        return results

    def _detect_from_shebang(
        self,
        content: str,
    ) -> ProgrammingLanguage | None:
        """Detect a language from the first-line shebang."""
        if not content:
            return None

        first_line = content.splitlines()[0] if content.splitlines() else ""

        match = self.SHEBANG_PATTERN.match(first_line)

        if match is None:
            return None

        executable = match.group("executable").lower()

        executable_path = Path(executable)
        executable_name = executable_path.name

        return self.SHEBANG_MAP.get(executable_name)