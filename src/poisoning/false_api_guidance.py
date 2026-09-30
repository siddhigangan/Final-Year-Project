"""False API guidance poisoning strategy."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import (
    PoisoningInput,
    PoisoningResult,
    PoisoningStrategy,
)


@dataclass(frozen=True)
class FalseApiGuidanceStrategy(PoisoningStrategy):
    """Controlled strategy for introducing incorrect API guidance."""

    category: str = "false_api_guidance"

    def apply(
        self,
        poisoning_input: PoisoningInput,
    ) -> PoisoningResult:
        """Apply false API guidance to a code chunk."""
        chunk = poisoning_input.chunk

        poisoned_content = self._build_poisoned_content(
            chunk=chunk,
            parameters=poisoning_input.parameters,
        )

        added_lines = self._added_line_count(
            poisoned_content,
            chunk.content,
        )

        guidance_type = self._guidance_type(
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
                "guidance_type": guidance_type,
            },
        )

        return PoisoningResult(
            original_chunk=chunk,
            poisoned_chunk=poisoned_chunk,
            category=self.category,
            poison_id=poisoning_input.poison_id,
            changed=poisoned_content != chunk.content,
            seed=poisoning_input.seed,
            description="Controlled false API guidance poisoning.",
            metadata={
                "language": chunk.language.value,
                "guidance_type": guidance_type,
                "marker": self._marker(
                    poisoning_input.parameters
                ),
            },
        )

    def _build_poisoned_content(
        self,
        *,
        chunk: CodeChunk,
        parameters: dict[str, object],
    ) -> str:
        """Build deterministic incorrect API guidance."""
        guidance_type = self._guidance_type(parameters)
        marker = self._marker(parameters)

        if chunk.language == ProgrammingLanguage.PYTHON:
            return self._build_python_content(
                content=chunk.content,
                guidance_type=guidance_type,
                marker=marker,
            )

        if chunk.language == ProgrammingLanguage.JAVA:
            return self._build_java_content(
                content=chunk.content,
                guidance_type=guidance_type,
                marker=marker,
            )

        return self._build_generic_content(
            content=chunk.content,
            guidance_type=guidance_type,
            marker=marker,
        )

    @staticmethod
    def _build_python_content(
        *,
        content: str,
        guidance_type: str,
        marker: str,
    ) -> str:
        """Append incorrect Python API guidance."""
        if guidance_type == "deprecated_method":
            guidance = (
                "# Incorrect API guidance: use "
                "legacy_method() instead of the current API "
                f"# {marker}"
            )
        elif guidance_type == "wrong_argument":
            guidance = (
                "# Incorrect API guidance: call api_method("
                "value, timeout='invalid') "
                f"# {marker}"
            )
        else:
            guidance = (
                "# Incorrect API guidance: "
                "use api_method_legacy() "
                f"# {marker}"
            )

        return FalseApiGuidanceStrategy._append_guidance(
            content,
            guidance,
        )

    @staticmethod
    def _build_java_content(
        *,
        content: str,
        guidance_type: str,
        marker: str,
    ) -> str:
        """Append incorrect Java API guidance."""
        if guidance_type == "deprecated_method":
            guidance = (
                "// Incorrect API guidance: use "
                "legacyMethod() instead of the current API "
                f"// {marker}"
            )
        elif guidance_type == "wrong_argument":
            guidance = (
                "// Incorrect API guidance: call "
                "apiMethod(value, \"invalid\") "
                f"// {marker}"
            )
        else:
            guidance = (
                "// Incorrect API guidance: use "
                "apiMethodLegacy() "
                f"// {marker}"
            )

        return FalseApiGuidanceStrategy._append_guidance(
            content,
            guidance,
        )

    @staticmethod
    def _build_generic_content(
        *,
        content: str,
        guidance_type: str,
        marker: str,
    ) -> str:
        """Append a generic incorrect API guidance marker."""
        guidance = (
            f"// false_api_guidance: "
            f"{guidance_type} - {marker}"
        )

        return FalseApiGuidanceStrategy._append_guidance(
            content,
            guidance,
        )

    @staticmethod
    def _append_guidance(
        content: str,
        guidance: str,
    ) -> str:
        """Append one deterministic guidance line."""
        if content.endswith("\n"):
            return f"{content}{guidance}\n"

        return f"{content}\n{guidance}\n"

    @staticmethod
    def _guidance_type(
        parameters: dict[str, object],
    ) -> str:
        """Read the requested guidance type."""
        value = parameters.get(
            "guidance_type",
            "deprecated_method",
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

        return "deprecated_method"

    @staticmethod
    def _marker(
        parameters: dict[str, object],
    ) -> str:
        """Read an optional custom marker."""
        value = parameters.get(
            "marker",
            "controlled false API guidance",
        )

        if isinstance(value, str) and value.strip():
            return value.strip()

        return "controlled false API guidance"

    @staticmethod
    def _added_line_count(
        poisoned_content: str,
        original_content: str,
    ) -> int:
        """Calculate the number of lines added."""
        return max(
            0,
            len(poisoned_content.splitlines())
            - len(original_content.splitlines()),
        )