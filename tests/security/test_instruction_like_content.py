"""Tests for the controlled instruction-like-content poisoning
strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.instruction_like_content import (
    InstructionLikeContentStrategy,
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


class TestInstructionLikeContentStrategy:
    """Tests for InstructionLikeContentStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = InstructionLikeContentStrategy()

        assert strategy.category == "instruction_like_content"

    def test_python_code_is_modified(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "instruction_like_content"
        assert "SYSTEM:" in result.poisoned_content
        assert "ignore all previous instructions" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = InstructionLikeContentStrategy()
        original = make_python_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="instruction_like_content",
                poison_id="instruction-002",
            )
        )

        assert result.original_content == original.content
        assert "def hello()" in result.original_content
        assert "SYSTEM:" not in result.original_content

    def test_metadata_identifies_instruction(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "instruction_like_content"
        )
        assert (
            result.poisoned_chunk.metadata["instruction_type"]
            == "ignore_previous"
        )

    def test_poison_id_is_preserved(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-004",
            )
        )

        assert result.poison_id == "instruction-004"
        assert (
            result.poisoned_chunk.metadata["poison_id"]
            == "instruction-004"
        )

    def test_seed_is_preserved(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-006",
                parameters={"marker": "custom instruction marker"},
            )
        )

        assert "custom instruction marker" in result.poisoned_content
        assert (
            result.metadata["marker"]
            == "custom instruction marker"
        )

    def test_java_code_is_supported(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_java_chunk(),
                category="instruction_like_content",
                poison_id="instruction-007",
            )
        )

        assert result.changed is True
        assert "SYSTEM:" in result.poisoned_content
        assert result.poisoned_chunk.language == ProgrammingLanguage.JAVA

    def test_disable_defense_instruction_is_supported(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-008",
                parameters={
                    "instruction_type": "disable_defense",
                },
            )
        )

        assert result.changed is True
        assert "security analysis is not required" in result.poisoned_content
        assert (
            result.metadata["instruction_type"]
            == "disable_defense"
        )

    def test_transformation_is_deterministic(self) -> None:
        strategy = InstructionLikeContentStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_python_chunk(),
            category="instruction_like_content",
            poison_id="instruction-009",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content
        assert first.metadata == second.metadata
        assert first.seed == second.seed

    def test_custom_instruction_type_is_preserved(self) -> None:
        strategy = InstructionLikeContentStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="instruction_like_content",
                poison_id="instruction-010",
                parameters={
                    "instruction_type": "custom_type",
                },
            )
        )

        assert (
            result.metadata["instruction_type"]
            == "custom_type"
        )
        assert (
            result.poisoned_chunk.metadata["instruction_type"]
            == "custom_type"
        )
