"""Tests for the controlled misleading-code poisoning strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.misleading_code import MisleadingCodeStrategy
from tests.security.test_poisoning import make_chunk


class TestMisleadingCodeStrategy:
    """Tests for MisleadingCodeStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = MisleadingCodeStrategy()

        assert strategy.category == "misleading_code"

    def test_python_code_is_modified(self) -> None:
        strategy = MisleadingCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="misleading_code",
                poison_id="misleading-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "misleading_code"
        assert "misleading_value" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = MisleadingCodeStrategy()

        original = make_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="misleading_code",
                poison_id="misleading-002",
            )
        )

        assert result.original_content == original.content
        assert result.poisoned_chunk.content != original.content

    def test_metadata_identifies_poisoning(self) -> None:
        strategy = MisleadingCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="misleading_code",
                poison_id="misleading-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "misleading_code"
        )
        assert result.poisoned_chunk.metadata["poison_id"] == "misleading-003"

    def test_poison_id_is_preserved(self) -> None:
        strategy = MisleadingCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="misleading_code",
                poison_id="misleading-004",
            )
        )

        assert result.poison_id == "misleading-004"

    def test_seed_is_preserved(self) -> None:
        strategy = MisleadingCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="misleading_code",
                poison_id="misleading-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = MisleadingCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_chunk(),
                category="misleading_code",
                poison_id="misleading-006",
                parameters={"marker": "custom marker"},
            )
        )

        assert "custom marker" in result.poisoned_content

    def test_java_code_is_supported(self) -> None:
        strategy = MisleadingCodeStrategy()

        chunk = CodeChunk(
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
    
        result = strategy(
            PoisoningInput(
                chunk=chunk,
                category="misleading_code",
                poison_id="misleading-007",
            )
        )

        assert result.changed is True
        assert "misleadingValue" in result.poisoned_content

    def test_transformation_is_deterministic(self) -> None:
        strategy = MisleadingCodeStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_chunk(),
            category="misleading_code",
            poison_id="misleading-008",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content