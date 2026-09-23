"""Tests for the controlled vulnerable-code poisoning strategy."""

from __future__ import annotations

from src.models import CodeChunk, ProgrammingLanguage
from src.poisoning.base import PoisoningInput
from src.poisoning.vulnerable_code import VulnerableCodeStrategy


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


class TestVulnerableCodeStrategy:
    """Tests for VulnerableCodeStrategy."""

    def test_strategy_has_expected_category(self) -> None:
        strategy = VulnerableCodeStrategy()

        assert strategy.category == "vulnerable_code"

    def test_python_code_is_modified(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-001",
            )
        )

        assert result.changed is True
        assert result.changed_content is True
        assert result.category == "vulnerable_code"
        assert "DEMO_API_KEY" in result.poisoned_content
        assert "DEMO-INSECURE-KEY" in result.poisoned_content

    def test_original_content_is_preserved(self) -> None:
        strategy = VulnerableCodeStrategy()
        original = make_python_chunk()

        result = strategy(
            PoisoningInput(
                chunk=original,
                category="vulnerable_code",
                poison_id="vulnerable-002",
            )
        )

        assert result.original_content == original.content
        assert "def hello()" in result.original_content
        assert "DEMO_API_KEY" not in result.original_content

    def test_metadata_identifies_vulnerability(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-003",
            )
        )

        assert result.poisoned_chunk.metadata["poisoned"] is True
        assert (
            result.poisoned_chunk.metadata["poison_category"]
            == "vulnerable_code"
        )
        assert (
            result.poisoned_chunk.metadata["vulnerability_type"]
            == "hardcoded_secret"
        )

    def test_poison_id_is_preserved(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-004",
            )
        )

        assert result.poison_id == "vulnerable-004"
        assert (
            result.poisoned_chunk.metadata["poison_id"]
            == "vulnerable-004"
        )

    def test_seed_is_preserved(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-005",
                seed=123,
            )
        )

        assert result.seed == 123

    def test_custom_marker_is_supported(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-006",
                parameters={"marker": "custom vulnerability marker"},
            )
        )

        assert "custom vulnerability marker" in result.poisoned_content
        assert (
            result.metadata["marker"]
            == "custom vulnerability marker"
        )

    def test_java_code_is_supported(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_java_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-007",
            )
        )

        assert result.changed is True
        assert "DEMO_API_KEY" in result.poisoned_content
        assert "DEMO-INSECURE-KEY" in result.poisoned_content
        assert result.poisoned_chunk.language == ProgrammingLanguage.JAVA

    def test_sql_injection_pattern_is_supported(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-008",
                parameters={
                    "vulnerability_type": "sql_injection",
                },
            )
        )

        assert result.changed is True
        assert "SELECT * FROM users" in result.poisoned_content
        assert (
            result.metadata["vulnerability_type"]
            == "sql_injection"
        )

    def test_transformation_is_deterministic(self) -> None:
        strategy = VulnerableCodeStrategy()

        poisoning_input = PoisoningInput(
            chunk=make_python_chunk(),
            category="vulnerable_code",
            poison_id="vulnerable-009",
            seed=42,
        )

        first = strategy(poisoning_input)
        second = strategy(poisoning_input)

        assert first.poisoned_content == second.poisoned_content
        assert first.metadata == second.metadata
        assert first.seed == second.seed

    def test_custom_vulnerability_type_is_preserved(self) -> None:
        strategy = VulnerableCodeStrategy()

        result = strategy(
            PoisoningInput(
                chunk=make_python_chunk(),
                category="vulnerable_code",
                poison_id="vulnerable-010",
                parameters={
                    "vulnerability_type": "custom_type",
                },
            )
        )

        assert (
            result.metadata["vulnerability_type"]
            == "custom_type"
        )
        assert (
            result.poisoned_chunk.metadata["vulnerability_type"]
            == "custom_type"
        )