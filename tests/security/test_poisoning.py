"""Tests for the controlled poisoning base abstractions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import (
    PoisoningInput,
    PoisoningInputError,
    PoisoningResult,
    PoisoningStrategy,
)


def make_chunk(
    *,
    content: str = "def hello():\n    return 'hello'\n",
) -> CodeChunk:
    return CodeChunk(
        chunk_id="chunk-1",
        source_file_id="file-1",
        repository_id="repo-1",
        content=content,
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=2,
        symbol_name="hello",
        symbol_type="function",
        is_documentation=False,
        metadata={},
    )


@dataclass(frozen=True)
class FakeStrategy(PoisoningStrategy):
    """Deterministic test strategy."""

    category: str = "test_poison"

    def apply(
        self,
        poisoning_input: PoisoningInput,
    ) -> PoisoningResult:
        poisoned = CodeChunk(
            chunk_id=poisoning_input.chunk.chunk_id,
            source_file_id=poisoning_input.chunk.source_file_id,
            repository_id=poisoning_input.chunk.repository_id,
            content=(
                poisoning_input.chunk.content
                + "\n# controlled poison\n"
            ),
            language=poisoning_input.chunk.language,
            start_line=poisoning_input.chunk.start_line,
            end_line=poisoning_input.chunk.end_line + 1,
            symbol_name=poisoning_input.chunk.symbol_name,
            symbol_type=poisoning_input.chunk.symbol_type,
            is_documentation=poisoning_input.chunk.is_documentation,
            metadata={
                **poisoning_input.chunk.metadata,
                "poisoned": True,
            },
        )

        return PoisoningResult(
            original_chunk=poisoning_input.chunk,
            poisoned_chunk=poisoned,
            category=self.category,
            poison_id=poisoning_input.poison_id,
            changed=True,
            seed=poisoning_input.seed,
            description="Controlled test poisoning.",
            metadata={"test": True},
        )


class TestPoisoningInput:
    """Validation tests for PoisoningInput."""

    def test_valid_input_is_accepted(self) -> None:
        value = PoisoningInput(
            chunk=make_chunk(),
            category="test_poison",
            poison_id="poison-001",
        )

        assert value.category == "test_poison"
        assert value.poison_id == "poison-001"
        assert value.seed == 42
        assert value.parameters == {}

    def test_invalid_chunk_is_rejected(self) -> None:
        with pytest.raises(PoisoningInputError):
            PoisoningInput(
                chunk="invalid",  # type: ignore[arg-type]
                category="test_poison",
                poison_id="poison-001",
            )

    def test_empty_category_is_rejected(self) -> None:
        with pytest.raises(PoisoningInputError):
            PoisoningInput(
                chunk=make_chunk(),
                category="",
                poison_id="poison-001",
            )

    def test_empty_poison_id_is_rejected(self) -> None:
        with pytest.raises(PoisoningInputError):
            PoisoningInput(
                chunk=make_chunk(),
                category="test_poison",
                poison_id="",
            )


class TestPoisoningStrategy:
    """Tests for the base strategy contract."""

    def test_strategy_accepts_valid_input(self) -> None:
        strategy = FakeStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_chunk(),
            category="test_poison",
            poison_id="poison-001",
        )

        result = strategy(poisoning_input)

        assert isinstance(result, PoisoningResult)
        assert result.changed is True
        assert result.changed_content is True
        assert result.poisoned_chunk.metadata["poisoned"] is True

    def test_category_mismatch_is_rejected(self) -> None:
        strategy = FakeStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_chunk(),
            category="wrong_category",
            poison_id="poison-001",
        )

        with pytest.raises(PoisoningInputError):
            strategy(poisoning_input)

    def test_unsupported_language_is_rejected(self) -> None:
        strategy = FakeStrategy()

        chunk = CodeChunk(
            chunk_id="chunk-1",
            source_file_id="file-1",
            repository_id="repo-1",
            content="puts 'hello'\n",
            language=ProgrammingLanguage.RUBY,
            start_line=1,
            end_line=1,
            symbol_name="hello",
            symbol_type="method",
            is_documentation=False,
            metadata={},
        )

        poisoning_input = PoisoningInput(
            chunk=chunk,
            category="test_poison",
            poison_id="poison-001",
        )

        with pytest.raises(PoisoningInputError):
            strategy(poisoning_input)

    def test_supports_language(self) -> None:
        strategy = FakeStrategy()

        assert strategy.supports_language(
            ProgrammingLanguage.PYTHON
        )
        assert strategy.supports_language(
            ProgrammingLanguage.JAVA
        )
        assert not strategy.supports_language(
            ProgrammingLanguage.RUBY
        )


class TestPoisoningResult:
    """Tests for PoisoningResult."""

    def test_serialization(self) -> None:
        strategy = FakeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="test_poison",
                poison_id="poison-001",
                seed=123,
            )
        )

        serialized = result.to_dict()

        assert serialized["poison_id"] == "poison-001"
        assert serialized["category"] == "test_poison"
        assert serialized["changed"] is True
        assert serialized["language"] == "python"
        assert serialized["metadata"]["test"] is True

    def test_original_and_poisoned_content_are_exposed(self) -> None:
        strategy = FakeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="test_poison",
                poison_id="poison-001",
            )
        )

        assert "return 'hello'" in result.original_content
        assert "controlled poison" in result.poisoned_content

    def test_invalid_metadata_is_rejected(self) -> None:
        with pytest.raises(PoisoningInputError):
            PoisoningResult(
                original_chunk=make_chunk(),
                poisoned_chunk=make_chunk(),
                category="test_poison",
                poison_id="poison-001",
                changed=False,
                seed=42,
                description="Test result.",
                metadata=[],  # type: ignore[arg-type]
            )


class TestPoisoningConfiguration:
    """Configuration-related tests for the poisoning strategy."""

    def test_default_configuration_is_available(self) -> None:
        strategy = FakeStrategy()

        assert strategy.category == "test_poison"

    def test_repr_contains_category(self) -> None:
        strategy = FakeStrategy()

        assert "test_poison" in repr(strategy)