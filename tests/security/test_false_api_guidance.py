"""Tests for the false API guidance poisoning strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.false_api_guidance import (
    FalseApiGuidanceStrategy,
)


def make_python_chunk() -> CodeChunk:
    """Create a Python test chunk."""
    return CodeChunk(
        chunk_id="chunk-python-1",
        source_file_id="file-python-1",
        repository_id="repo-1",
        content="def hello():\n    return 'hello'\n",
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


class TestFalseApiGuidanceStrategy:
    """Tests for FalseApiGuidanceStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        assert strategy.category == "false_api_guidance"

    def test_python_code_is_modified(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "false_api_guidance"
        assert "Incorrect API guidance" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = FalseApiGuidanceStrategy()
        original = make_python_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="false_api_guidance",
                poison_id="api-002",
            )
        )

        assert result.original_content == original.content
        assert "Incorrect API guidance" not in result.original_content

    def test_metadata_identifies_guidance(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "false_api_guidance"
        )
        assert (
            result.poisoned_chunk.metadata["guidance_type"]
            == "deprecated_method"
        )

    def test_poison_id_is_preserved(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-004",
            )
        )

        assert result.poison_id == "api-004"
        assert (
            result.poisoned_chunk.metadata["poison_id"]
            == "api-004"
        )

    def test_seed_is_preserved(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-006",
                parameters={"marker": "custom API marker"},
            )
        )

        assert "custom API marker" in result.poisoned_content
        assert (
            result.metadata["marker"]
            == "custom API marker"
        )

    def test_java_code_is_supported(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_java_chunk(),
                category="false_api_guidance",
                poison_id="api-007",
            )
        )

        assert result.changed is True
        assert "Incorrect API guidance" in result.poisoned_content
        assert "legacyMethod" in result.poisoned_content
        assert (
            result.poisoned_chunk.language
            == ProgrammingLanguage.JAVA
        )

    def test_wrong_argument_guidance_is_supported(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-008",
                parameters={
                    "guidance_type": "wrong_argument",
                },
            )
        )

        assert result.changed is True
        assert "wrong_argument" in result.metadata[
            "guidance_type"
        ]
        assert "invalid" in result.poisoned_content

    def test_custom_guidance_type_is_preserved(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="false_api_guidance",
                poison_id="api-009",
                parameters={
                    "guidance_type": "custom_guidance",
                },
            )
        )

        assert (
            result.metadata["guidance_type"]
            == "custom_guidance"
        )
        assert (
            result.poisoned_chunk.metadata["guidance_type"]
            == "custom_guidance"
        )

    def test_transformation_is_deterministic(self) -> None:
        strategy = FalseApiGuidanceStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_python_chunk(),
            category="false_api_guidance",
            poison_id="api-010",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content
        assert first.metadata == second.metadata
        assert first.seed == second.seed