"""Tests for the controlled contradictory-documentation poisoning
strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.contradictory_documentation import (
    ContradictoryDocumentationStrategy,
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


class TestContradictoryDocumentationStrategy:
    """Tests for ContradictoryDocumentationStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        assert strategy.category == "contradictory_documentation"

    def test_python_code_is_modified(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "contradictory_documentation"
        assert "NOTE:" in result.poisoned_content
        assert "always returns None" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = ContradictoryDocumentationStrategy()
        original = make_python_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="contradictory_documentation",
                poison_id="contradiction-002",
            )
        )

        assert result.original_content == original.content
        assert "def hello()" in result.original_content
        assert "NOTE:" not in result.original_content

    def test_metadata_identifies_contradiction(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "contradictory_documentation"
        )
        assert (
            result.poisoned_chunk.metadata["contradiction_type"]
            == "return_value"
        )

    def test_poison_id_is_preserved(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-004",
            )
        )

        assert result.poison_id == "contradiction-004"
        assert (
            result.poisoned_chunk.metadata["poison_id"]
            == "contradiction-004"
        )

    def test_seed_is_preserved(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-006",
                parameters={"marker": "custom contradiction marker"},
            )
        )

        assert "custom contradiction marker" in result.poisoned_content
        assert (
            result.metadata["marker"]
            == "custom contradiction marker"
        )

    def test_java_code_is_supported(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_java_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-007",
            )
        )

        assert result.changed is True
        assert "NOTE:" in result.poisoned_content
        assert "always returns null" in result.poisoned_content
        assert result.poisoned_chunk.language == ProgrammingLanguage.JAVA

    def test_side_effect_contradiction_is_supported(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-008",
                parameters={
                    "contradiction_type": "side_effect",
                },
            )
        )

        assert result.changed is True
        assert "no side effects" in result.poisoned_content
        assert (
            result.metadata["contradiction_type"]
            == "side_effect"
        )

    def test_transformation_is_deterministic(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_python_chunk(),
            category="contradictory_documentation",
            poison_id="contradiction-009",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content
        assert first.metadata == second.metadata
        assert first.seed == second.seed

    def test_custom_contradiction_type_is_preserved(self) -> None:
        strategy = ContradictoryDocumentationStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="contradictory_documentation",
                poison_id="contradiction-010",
                parameters={
                    "contradiction_type": "custom_type",
                },
            )
        )

        assert (
            result.metadata["contradiction_type"]
            == "custom_type"
        )
        assert (
            result.poisoned_chunk.metadata["contradiction_type"]
            == "custom_type"
        )
