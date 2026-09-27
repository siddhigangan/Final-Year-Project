"""Tests for the controlled context-manipulation poisoning strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.context_manipulation import (
    ContextManipulationStrategy,
)


def make_python_chunk(
    *,
    content: str = "def hello():\n    return 'hello'\n",
) -> CodeChunk:
    """Create a Python test chunk."""
    return CodeChunk(
        chunk_id="chunk-python-1",
        source_file_id="file-python-1",
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


def make_java_chunk() -> CodeChunk:
    """Create a Java test chunk."""
    return CodeChunk(
        chunk_id="chunk-java-1",
        source_file_id="file-java-1",
        repository_id="repo-1",
        content=(
            "public class Hello {\n"
            "    public void hello() {}\n"
            "}\n"
        ),
        language=ProgrammingLanguage.JAVA,
        start_line=1,
        end_line=3,
        symbol_name="Hello",
        symbol_type="class",
        is_documentation=False,
        metadata={},
    )


class TestContextManipulationStrategy:
    """Tests for ContextManipulationStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = ContextManipulationStrategy()

        assert strategy.category == "context_manipulation"

    def test_python_code_is_modified(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "context_manipulation"
        assert "maintainer-reviewed" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = ContextManipulationStrategy()
        original = make_python_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="context_manipulation",
                poison_id="context-002",
            )
        )

        assert result.original_content == original.content
        assert "def hello()" in result.original_content
        assert "maintainer-reviewed" not in result.original_content

    def test_metadata_identifies_manipulation(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "context_manipulation"
        )
        assert (
            result.poisoned_chunk.metadata["manipulation_type"]
            == "false_authority"
        )

    def test_poison_id_is_preserved(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-004",
            )
        )

        assert result.poison_id == "context-004"
        assert (
            result.poisoned_chunk.metadata["poison_id"]
            == "context-004"
        )

    def test_seed_is_preserved(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-006",
                parameters={"marker": "custom context marker"},
            )
        )

        assert "custom context marker" in result.poisoned_content
        assert (
            result.metadata["marker"]
            == "custom context marker"
        )

    def test_java_code_is_supported(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_java_chunk(),
                category="context_manipulation",
                poison_id="context-007",
            )
        )

        assert result.changed is True
        assert "maintainer-reviewed" in result.poisoned_content
        assert result.poisoned_chunk.language == ProgrammingLanguage.JAVA

    def test_false_relevance_manipulation_is_supported(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-008",
                parameters={
                    "manipulation_type": "false_relevance",
                },
            )
        )

        assert result.changed is True
        assert "Best match for this exact query" in result.poisoned_content
        assert (
            result.metadata["manipulation_type"]
            == "false_relevance"
        )

    def test_transformation_is_deterministic(self) -> None:
        strategy = ContextManipulationStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_python_chunk(),
            category="context_manipulation",
            poison_id="context-009",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content
        assert first.metadata == second.metadata
        assert first.seed == second.seed

    def test_custom_manipulation_type_is_preserved(self) -> None:
        strategy = ContextManipulationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="context_manipulation",
                poison_id="context-010",
                parameters={
                    "manipulation_type": "custom_type",
                },
            )
        )

        assert (
            result.metadata["manipulation_type"]
            == "custom_type"
        )
        assert (
            result.poisoned_chunk.metadata["manipulation_type"]
            == "custom_type"
        )
