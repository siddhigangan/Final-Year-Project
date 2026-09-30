"""Base abstractions for controlled SecureCodeRAG poisoning strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.models import CodeChunk, ProgrammingLanguage


class PoisoningError(RuntimeError):
    """Base exception for poisoning-related failures."""


class PoisoningInputError(PoisoningError):
    """Raised when poisoning input is invalid."""


class PoisoningConfigurationError(PoisoningError):
    """Raised when a poisoning strategy is incorrectly configured."""


@dataclass(frozen=True)
class PoisoningInput:
    """Input describing one controlled poisoning operation."""

    chunk: CodeChunk
    category: str
    poison_id: str
    seed: int = 42
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, CodeChunk):
            raise PoisoningInputError(
                "chunk must be a CodeChunk."
            )

        if not isinstance(self.category, str):
            raise PoisoningInputError(
                "category must be a string."
            )

        if not self.category.strip():
            raise PoisoningInputError(
                "category cannot be empty."
            )

        if not isinstance(self.poison_id, str):
            raise PoisoningInputError(
                "poison_id must be a string."
            )

        if not self.poison_id.strip():
            raise PoisoningInputError(
                "poison_id cannot be empty."
            )

        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
        ):
            raise PoisoningInputError(
                "seed must be an integer."
            )

        if not isinstance(self.parameters, dict):
            raise PoisoningInputError(
                "parameters must be a dictionary."
            )


@dataclass(frozen=True)
class PoisoningResult:
    """Result of one controlled poisoning transformation."""

    original_chunk: CodeChunk
    poisoned_chunk: CodeChunk
    category: str
    poison_id: str
    changed: bool
    seed: int
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.original_chunk, CodeChunk):
            raise PoisoningInputError(
                "original_chunk must be a CodeChunk."
            )

        if not isinstance(self.poisoned_chunk, CodeChunk):
            raise PoisoningInputError(
                "poisoned_chunk must be a CodeChunk."
            )

        if not isinstance(self.category, str):
            raise PoisoningInputError(
                "category must be a string."
            )

        if not self.category.strip():
            raise PoisoningInputError(
                "category cannot be empty."
            )

        if not isinstance(self.poison_id, str):
            raise PoisoningInputError(
                "poison_id must be a string."
            )

        if not self.poison_id.strip():
            raise PoisoningInputError(
                "poison_id cannot be empty."
            )

        if not isinstance(self.changed, bool):
            raise PoisoningInputError(
                "changed must be a boolean."
            )

        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
        ):
            raise PoisoningInputError(
                "seed must be an integer."
            )

        if not isinstance(self.description, str):
            raise PoisoningInputError(
                "description must be a string."
            )

        if not self.description.strip():
            raise PoisoningInputError(
                "description cannot be empty."
            )

        if not isinstance(self.metadata, dict):
            raise PoisoningInputError(
                "metadata must be a dictionary."
            )

    @property
    def original_content(self) -> str:
        """Return the original chunk content."""
        return self.original_chunk.content

    @property
    def poisoned_content(self) -> str:
        """Return the resulting poisoned content."""
        return self.poisoned_chunk.content

    @property
    def changed_content(self) -> bool:
        """Return whether the content itself changed."""
        return self.original_content != self.poisoned_content

    def to_dict(self) -> dict[str, Any]:
        """Serialize the poisoning result."""
        return {
            "poison_id": self.poison_id,
            "category": self.category,
            "changed": self.changed,
            "changed_content": self.changed_content,
            "seed": self.seed,
            "description": self.description,
            "original_chunk_id": self.original_chunk.chunk_id,
            "poisoned_chunk_id": self.poisoned_chunk.chunk_id,
            "source_file_id": self.poisoned_chunk.source_file_id,
            "repository_id": self.poisoned_chunk.repository_id,
            "language": self.poisoned_chunk.language.value,
            "metadata": dict(self.metadata),
        }


class PoisoningStrategy(ABC):
    """Abstract interface for one controlled poisoning category."""

    category: str = "unknown"
    supported_languages: tuple[ProgrammingLanguage, ...] = (
        ProgrammingLanguage.PYTHON,
        ProgrammingLanguage.JAVASCRIPT,
        ProgrammingLanguage.TYPESCRIPT,
        ProgrammingLanguage.JAVA,
        ProgrammingLanguage.CPP,
    )

    def __init__(
        self,
        *,
        seed: int = 42,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if (
            not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise PoisoningConfigurationError(
                "seed must be an integer."
            )

        if not isinstance(self.category, str):
            raise PoisoningConfigurationError(
                "category must be a string."
            )

        if not self.category.strip():
            raise PoisoningConfigurationError(
                "category cannot be empty."
            )

        if not isinstance(self.supported_languages, tuple):
            raise PoisoningConfigurationError(
                "supported_languages must be a tuple."
            )

        if any(
            not isinstance(language, ProgrammingLanguage)
            for language in self.supported_languages
        ):
            raise PoisoningConfigurationError(
                "supported_languages must contain "
                "ProgrammingLanguage values."
            )

        if parameters is not None and not isinstance(parameters, dict):
            raise PoisoningConfigurationError(
                "parameters must be a dictionary or None."
            )

        self.seed = seed
        self.parameters = dict(parameters or {})

    @property
    def name(self) -> str:
        """Return the stable strategy name."""
        return self.category

    def supports_language(
        self,
        language: ProgrammingLanguage,
    ) -> bool:
        """Return whether this strategy supports the language."""
        if not isinstance(language, ProgrammingLanguage):
            raise PoisoningInputError(
                "language must be a ProgrammingLanguage."
            )

        return language in self.supported_languages

    def validate_input(
        self,
        poisoning_input: PoisoningInput,
    ) -> None:
        """Validate that input belongs to this strategy."""
        if not isinstance(poisoning_input, PoisoningInput):
            raise PoisoningInputError(
                "poisoning_input must be a PoisoningInput."
            )

        if poisoning_input.category != self.category:
            raise PoisoningInputError(
                "Poisoning input category does not match "
                f"strategy category {self.category!r}."
            )

        if not self.supports_language(
            poisoning_input.chunk.language
        ):
            raise PoisoningInputError(
                f"Language {poisoning_input.chunk.language.value!r} "
                f"is not supported by strategy {self.category!r}."
            )

    @abstractmethod
    def apply(
        self,
        poisoning_input: PoisoningInput,
    ) -> PoisoningResult:
        """Apply the controlled poisoning transformation."""

    def __call__(
        self,
        poisoning_input: PoisoningInput,
    ) -> PoisoningResult:
        """Validate and apply the poisoning strategy."""
        self.validate_input(poisoning_input)
        return self.apply(poisoning_input)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"category={self.category!r}, "
            f"seed={self.seed})"
        )