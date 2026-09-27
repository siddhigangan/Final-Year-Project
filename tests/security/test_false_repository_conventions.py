"""Tests for the controlled false-repository-conventions poisoning
strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.false_repository_conventions import (
    FalseRepositoryConventionsStrategy,
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


class TestFalseRepositoryConventionsStrategy:
    """Tests for FalseRepositoryConventionsStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        assert strategy.category == "false_repository_conventions"

    def test_python_code_is_modified(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "false_repository_conventions"
        assert "PROJECT CONVENTION:" in result.poisoned_content
        assert "do not add validation here" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()
        original = make_python_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="false_repository_conventions",
                poison_id="convention-002",
            )
        )

        assert result.original_content == original.content
        assert "def hello()" in result.original_content
        assert "PROJECT CONVENTION:" not in result.original_content

    def test_metadata_identifies_convention(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "false_repository_conventions"
        )
        assert (
            result.poisoned_chunk.metadata["convention_type"]
            == "disable_validation"
        )

    def test_poison_id_is_preserved(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-004",
            )
        )

        assert result.poison_id == "convention-004"
        assert (
            result.poisoned_chunk.metadata["poison_id"]
            == "convention-004"
        )

    def test_seed_is_preserved(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-006",
                parameters={"marker": "custom convention marker"},
            )
        )

        assert "custom convention marker" in result.poisoned_content
        assert (
            result.metadata["marker"]
            == "custom convention marker"
        )

    def test_java_code_is_supported(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_java_chunk(),
                category="false_repository_conventions",
                poison_id="convention-007",
            )
        )

        assert result.changed is True
        assert "PROJECT CONVENTION:" in result.poisoned_content
        assert result.poisoned_chunk.language == ProgrammingLanguage.JAVA

    def test_shared_credentials_convention_is_supported(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-008",
                parameters={
                    "convention_type": "shared_credentials",
                },
            )
        )

        assert result.changed is True
        assert "DEV_CREDENTIALS" in result.poisoned_content
        assert (
            result.metadata["convention_type"]
            == "shared_credentials"
        )

    def test_transformation_is_deterministic(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_python_chunk(),
            category="false_repository_conventions",
            poison_id="convention-009",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content
        assert first.metadata == second.metadata
        assert first.seed == second.seed

    def test_custom_convention_type_is_preserved(self) -> None:
        strategy = FalseRepositoryConventionsStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_repository_conventions",
                poison_id="convention-010",
                parameters={
                    "convention_type": "custom_type",
                },
            )
        )

        assert (
            result.metadata["convention_type"]
            == "custom_type"
        )
        assert (
            result.poisoned_chunk.metadata["convention_type"]
            == "custom_type"
        )
