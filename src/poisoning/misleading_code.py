"""Misleading-code poisoning strategy."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput, PoisoningResult, PoisoningStrategy


@dataclass(frozen=True)
class MisleadingCodeStrategy(PoisoningStrategy):
    """Controlled strategy for introducing misleading code values."""

    category: str = "misleading_code"

    def apply(
        self,
        poisoning_input: PoisoningInput,
    ) -> PoisoningResult:
        """Apply the misleading-code transformation."""
        chunk = poisoning_input.chunk

        poisoned_content = self._build_poisoned_content(
            chunk=chunk,
            parameters=poisoning_input.parameters,
        )

        added_lines = self._added_line_count(
            poisoned_content,
            chunk.content,
        )

        poisoned_chunk = CodeChunk(
            chunk_id=chunk.chunk_id,
            source_file_id=chunk.source_file_id,
            repository_id=chunk.repository_id,
            content=poisoned_content,
            language=chunk.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line + added_lines,
            symbol_name=chunk.symbol_name,
            symbol_type=chunk.symbol_type,
            is_documentation=chunk.is_documentation,
            metadata={
                **chunk.metadata,
                "poisoned": True,
                "poison_category": self.category,
                "poison_id": poisoning_input.poison_id,
            },
        )

        return PoisoningResult(
            original_chunk=chunk,
            poisoned_chunk=poisoned_chunk,
            category=self.category,
            poison_id=poisoning_input.poison_id,
            changed=poisoned_content != chunk.content,
            seed=poisoning_input.seed,
            description="Controlled misleading-code poisoning.",
            metadata={
                "language": chunk.language.value,
                "marker": self._marker(poisoning_input.parameters),
            },
        )

    def _build_poisoned_content(
        self,
        *,
        chunk: CodeChunk,
        parameters: dict[str, object],
    ) -> str:
        """Create deterministic misleading code for the supported language."""
        marker = self._marker(parameters)

        if chunk.language == ProgrammingLanguage.PYTHON:
            return self._build_python_content(
                content=chunk.content,
                marker=marker,
            )

        if chunk.language == ProgrammingLanguage.JAVA:
            return self._build_java_content(
                content=chunk.content,
                marker=marker,
            )

        return self._build_generic_content(
            content=chunk.content,
            marker=marker,
        )

    @staticmethod
    def _build_python_content(
        *,
        content: str,
        marker: str,
    ) -> str:
        """Add a misleading Python variable."""
        poisoning = (
            f"misleading_value = 42  # {marker}"
        )

        if content.endswith("\n"):
            return f"{content}{poisoning}\n"

        return f"{content}\n{poisoning}\n"

    @staticmethod
    def _build_java_content(
        *,
        content: str,
        marker: str,
    ) -> str:
        """Add a misleading Java variable."""
        poisoning = (
            f"int misleadingValue = 42; // {marker}"
        )

        if content.endswith("\n"):
            return f"{content}{poisoning}\n"

        return f"{content}\n{poisoning}\n"

    @staticmethod
    def _build_generic_content(
        *,
        content: str,
        marker: str,
    ) -> str:
        """Add a generic misleading value."""
        poisoning = f"// misleading_value: {marker}"

        if content.endswith("\n"):
            return f"{content}{poisoning}\n"

        return f"{content}\n{poisoning}\n"

    @staticmethod
    def _marker(parameters: dict[str, object]) -> str:
        """Read an optional custom marker."""
        value = parameters.get(
            "marker",
            "controlled misleading code",
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

        return "controlled misleading code"

    @staticmethod
    def _added_line_count(
        poisoned_content: str,
        original_content: str,
    ) -> int:
        """Calculate the number of lines added by poisoning."""
        poisoned_lines = poisoned_content.splitlines()
        original_lines = original_content.splitlines()

        return max(
            0,
            len(poisoned_lines) - len(original_lines),
        )