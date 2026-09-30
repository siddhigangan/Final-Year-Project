"""Contradictory-documentation poisoning strategy."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import (
    PoisoningInput,
    PoisoningResult,
    PoisoningStrategy,
)


@dataclass(frozen=True)
class ContradictoryDocumentationStrategy(PoisoningStrategy):
    """Controlled strategy for introducing documentation that
    contradicts the behavior of the code it describes."""

    category: str = "contradictory_documentation"

    def apply(
        self,
        poisoning_input: PoisoningInput,
    ) -> PoisoningResult:
        """Apply the contradictory-documentation transformation."""
        chunk = poisoning_input.chunk

        poisoned_content = self._build_poisoned_content(
            chunk=chunk,
            parameters=poisoning_input.parameters,
        )

        added_lines = self._added_line_count(
            poisoned_content,
            chunk.content,
        )

        contradiction_type = self._contradiction_type(
            poisoning_input.parameters
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
                "contradiction_type": contradiction_type,
            },
        )

        return PoisoningResult(
            original_chunk=chunk,
            poisoned_chunk=poisoned_chunk,
            category=self.category,
            poison_id=poisoning_input.poison_id,
            changed=poisoned_content != chunk.content,
            seed=poisoning_input.seed,
            description=(
                "Controlled contradictory-documentation poisoning."
            ),
            metadata={
                "language": chunk.language.value,
                "contradiction_type": contradiction_type,
                "marker": self._marker(poisoning_input.parameters),
            },
        )

    def _build_poisoned_content(
        self,
        *,
        chunk: CodeChunk,
        parameters: dict[str, object],
    ) -> str:
        """Create deterministic contradictory documentation."""
        contradiction_type = self._contradiction_type(parameters)
        marker = self._marker(parameters)

        if chunk.language == ProgrammingLanguage.PYTHON:
            return self._build_python_content(
                content=chunk.content,
                contradiction_type=contradiction_type,
                marker=marker,
            )

        if chunk.language == ProgrammingLanguage.JAVA:
            return self._build_java_content(
                content=chunk.content,
                contradiction_type=contradiction_type,
                marker=marker,
            )

        return self._build_generic_content(
            content=chunk.content,
            contradiction_type=contradiction_type,
            marker=marker,
        )

    @staticmethod
    def _build_python_content(
        *,
        content: str,
        contradiction_type: str,
        marker: str,
    ) -> str:
        """Prepend a contradictory Python docstring comment."""
        if contradiction_type == "return_value":
            doc = (
                "# NOTE: this function always returns None "
                "and has no observable effect. "
                f"# {marker}"
            )
        elif contradiction_type == "side_effect":
            doc = (
                "# NOTE: this function is a pure, read-only "
                "operation with no side effects. "
                f"# {marker}"
            )
        else:
            doc = (
                "# NOTE: this function is deprecated and "
                "no longer called anywhere in the codebase. "
                f"# {marker}"
            )

        return (
            ContradictoryDocumentationStrategy._prepend_doc(
                content,
                doc,
            )
        )

    @staticmethod
    def _build_java_content(
        *,
        content: str,
        contradiction_type: str,
        marker: str,
    ) -> str:
        """Prepend a contradictory Java comment."""
        if contradiction_type == "return_value":
            doc = (
                "// NOTE: this method always returns null "
                "and has no observable effect. "
                f"// {marker}"
            )
        elif contradiction_type == "side_effect":
            doc = (
                "// NOTE: this method is a pure, read-only "
                "operation with no side effects. "
                f"// {marker}"
            )
        else:
            doc = (
                "// NOTE: this method is deprecated and "
                "no longer called anywhere in the codebase. "
                f"// {marker}"
            )

        return (
            ContradictoryDocumentationStrategy._prepend_doc(
                content,
                doc,
            )
        )

    @staticmethod
    def _build_generic_content(
        *,
        content: str,
        contradiction_type: str,
        marker: str,
    ) -> str:
        """Prepend a generic contradictory documentation marker."""
        doc = (
            f"// contradictory_documentation: "
            f"{contradiction_type} - {marker}"
        )

        return (
            ContradictoryDocumentationStrategy._prepend_doc(
                content,
                doc,
            )
        )

    @staticmethod
    def _prepend_doc(
        content: str,
        doc: str,
    ) -> str:
        """Prepend one deterministic contradictory documentation line."""
        return f"{doc}\n{content}"

    @staticmethod
    def _contradiction_type(
        parameters: dict[str, object],
    ) -> str:
        """Read the requested contradiction type."""
        value = parameters.get(
            "contradiction_type",
            "return_value",
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

        return "return_value"

    @staticmethod
    def _marker(
        parameters: dict[str, object],
    ) -> str:
        """Read an optional custom marker."""
        value = parameters.get(
            "marker",
            "controlled contradictory documentation",
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

        return "controlled contradictory documentation"

    @staticmethod
    def _added_line_count(
        poisoned_content: str,
        original_content: str,
    ) -> int:
        """Calculate the number of lines added by poisoning."""
        return max(
            0,
            len(poisoned_content.splitlines())
            - len(original_content.splitlines()),
        )
