"""Unit tests for the SecureCodeRAG AST parser."""

from __future__ import annotations

import pytest

from src.models import ProgrammingLanguage
from src.parsing.ast_parser import (
    ASTParser,
    ASTParserError,
    UnsupportedASTLanguageError,
)

PYTHON_SOURCE = """\
def add(a, b):
    return a + b
"""


JAVASCRIPT_SOURCE = """\
function add(a, b) {
    return a + b;
}
"""


TYPESCRIPT_SOURCE = """\
interface User {
    id: number;
    name: string;
}

const user: User = {
    id: 1,
    name: "Soham",
};
"""


JAVA_SOURCE = """\
class Calculator {
    int add(int a, int b) {
        return a + b;
    }
}
"""


C_SOURCE = """\
int add(int a, int b) {
    return a + b;
}
"""


CPP_SOURCE = """\
class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }
};
"""


def test_parser_initializes() -> None:
    """ASTParser should initialize successfully."""
    parser = ASTParser()

    assert parser is not None


def test_python_source_parses_without_errors() -> None:
    """Valid Python source should produce a clean AST."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    assert result.language is ProgrammingLanguage.PYTHON
    assert result.root_node_type == "module"
    assert result.has_errors is False
    assert result.error_node_count == 0
    assert result.node_count > 1


def test_javascript_source_parses_without_errors() -> None:
    """Valid JavaScript source should produce a clean AST."""
    parser = ASTParser()

    result = parser.parse(
        JAVASCRIPT_SOURCE,
        ProgrammingLanguage.JAVASCRIPT,
    )

    assert result.language is ProgrammingLanguage.JAVASCRIPT
    assert result.root_node_type == "program"
    assert result.has_errors is False
    assert result.error_node_count == 0
    assert result.node_count > 1


def test_typescript_source_parses_without_errors() -> None:
    """TypeScript-specific syntax should parse without syntax errors."""
    parser = ASTParser()

    result = parser.parse(
        TYPESCRIPT_SOURCE,
        ProgrammingLanguage.TYPESCRIPT,
    )

    assert result.language is ProgrammingLanguage.TYPESCRIPT
    assert result.has_errors is False
    assert result.error_node_count == 0
    assert result.node_count > 1


def test_java_source_parses_without_errors() -> None:
    """Valid Java source should produce a clean AST."""
    parser = ASTParser()

    result = parser.parse(
        JAVA_SOURCE,
        ProgrammingLanguage.JAVA,
    )

    assert result.language is ProgrammingLanguage.JAVA
    assert result.root_node_type == "program"
    assert result.has_errors is False
    assert result.error_node_count == 0
    assert result.node_count > 1


def test_c_source_parses_without_errors() -> None:
    """Valid C source should produce a clean AST."""
    parser = ASTParser()

    result = parser.parse(
        C_SOURCE,
        ProgrammingLanguage.C,
    )

    assert result.language is ProgrammingLanguage.C
    assert result.has_errors is False
    assert result.error_node_count == 0
    assert result.node_count > 1


def test_cpp_source_parses_without_errors() -> None:
    """Valid C++ source should produce a clean AST."""
    parser = ASTParser()

    result = parser.parse(
        CPP_SOURCE,
        ProgrammingLanguage.CPP,
    )

    assert result.language is ProgrammingLanguage.CPP
    assert result.root_node_type == "translation_unit"
    assert result.has_errors is False
    assert result.error_node_count == 0
    assert result.node_count > 1


def test_invalid_python_source_reports_errors() -> None:
    """Malformed source should be reported as syntactically invalid."""
    parser = ASTParser()

    source = """\
def broken(
    return 42
"""

    result = parser.parse(
        source,
        ProgrammingLanguage.PYTHON,
    )

    assert result.has_errors is True
    assert result.error_node_count > 0
    assert result.node_count > 1


def test_empty_source_is_rejected() -> None:
    """Empty source should raise an ASTParserError."""
    parser = ASTParser()

    with pytest.raises(ASTParserError, match="source must not be empty"):
        parser.parse(
            "",
            ProgrammingLanguage.PYTHON,
        )


def test_non_string_source_is_rejected() -> None:
    """Non-string source should raise an ASTParserError."""
    parser = ASTParser()

    with pytest.raises(
        ASTParserError,
        match="source must be a string",
    ):
        parser.parse(
            123,
            ProgrammingLanguage.PYTHON,
        )


def test_invalid_language_type_is_rejected() -> None:
    """A non-ProgrammingLanguage value should be rejected."""
    parser = ASTParser()

    with pytest.raises(
        ASTParserError,
        match="language must be a ProgrammingLanguage value",
    ):
        parser.parse(
            PYTHON_SOURCE,
            "python",
        )


def test_unsupported_language_raises_error() -> None:
    """Languages without registered grammars should be rejected."""
    parser = ASTParser()

    with pytest.raises(UnsupportedASTLanguageError):
        parser.parse(
            "package main\nfunc main() {}\n",
            ProgrammingLanguage.GO,
        )


def test_parse_bytes_matches_string_parsing() -> None:
    """UTF-8 bytes should produce the same AST information."""
    parser = ASTParser()

    string_result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    bytes_result = parser.parse_bytes(
        PYTHON_SOURCE.encode("utf-8"),
        ProgrammingLanguage.PYTHON,
    )

    assert bytes_result.language is string_result.language
    assert bytes_result.root_node_type == string_result.root_node_type
    assert bytes_result.has_errors == string_result.has_errors
    assert bytes_result.error_node_count == string_result.error_node_count
    assert bytes_result.node_count == string_result.node_count


def test_invalid_utf8_bytes_are_rejected() -> None:
    """Invalid UTF-8 bytes should raise an ASTParserError."""
    parser = ASTParser()

    with pytest.raises(
        ASTParserError,
        match="source bytes must contain valid UTF-8",
    ):
        parser.parse_bytes(
            b"\xff\xfe\xfd",
            ProgrammingLanguage.PYTHON,
        )


def test_non_bytes_input_is_rejected() -> None:
    """Non-bytes input should be rejected by parse_bytes."""
    parser = ASTParser()

    with pytest.raises(
        ASTParserError,
        match="source must be bytes",
    ):
        parser.parse_bytes(
            "not bytes",
            ProgrammingLanguage.PYTHON,
        )


def test_parser_for_returns_cached_parser() -> None:
    """Repeated parser requests should reuse the same parser."""
    parser = ASTParser()

    first = parser.parser_for(ProgrammingLanguage.PYTHON)
    second = parser.parser_for(ProgrammingLanguage.PYTHON)

    assert first is second


def test_different_languages_have_different_parsers() -> None:
    """Different languages should use separate parser instances."""
    parser = ASTParser()

    python_parser = parser.parser_for(
        ProgrammingLanguage.PYTHON,
    )
    javascript_parser = parser.parser_for(
        ProgrammingLanguage.JAVASCRIPT,
    )

    assert python_parser is not javascript_parser


def test_clear_cache_removes_cached_parsers() -> None:
    """clear_cache should remove cached parser instances."""
    parser = ASTParser()

    first = parser.parser_for(ProgrammingLanguage.PYTHON)

    parser.clear_cache()

    second = parser.parser_for(ProgrammingLanguage.PYTHON)

    assert first is not second


@pytest.mark.parametrize(
    "language",
    [
        ProgrammingLanguage.PYTHON,
        ProgrammingLanguage.JAVASCRIPT,
        ProgrammingLanguage.TYPESCRIPT,
        ProgrammingLanguage.JAVA,
        ProgrammingLanguage.C,
        ProgrammingLanguage.CPP,
    ],
)
def test_supported_languages_report_true(
    language: ProgrammingLanguage,
) -> None:
    """Registered AST languages should be reported as supported."""
    parser = ASTParser()

    assert parser.is_supported(language) is True


def test_unsupported_language_reports_false() -> None:
    """An unregistered language should be reported as unsupported."""
    parser = ASTParser()

    assert parser.is_supported(
        ProgrammingLanguage.GO,
    ) is False


def test_walk_visits_every_node() -> None:
    """walk should visit the same number of nodes counted by the parser."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    nodes = parser.walk(result.tree.root_node)

    assert len(nodes) == result.node_count
    assert nodes[0].type == result.tree.root_node.type
    assert nodes[0].start_byte == result.tree.root_node.start_byte
    assert nodes[0].end_byte == result.tree.root_node.end_byte


def test_walk_returns_depth_first_order() -> None:
    """walk should start with the root node."""
    parser = ASTParser()

    result = parser.parse(
        JAVASCRIPT_SOURCE,
        ProgrammingLanguage.JAVASCRIPT,
    )

    nodes = parser.walk(result.tree.root_node)

    assert nodes
    assert nodes[0].type == result.root_node_type


def test_node_text_returns_exact_source_text() -> None:
    """node_text should recover the source represented by a node."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    root = result.tree.root_node

    text = parser.node_text(
        root,
        PYTHON_SOURCE,
    )

    assert text == PYTHON_SOURCE


def test_node_text_returns_function_text() -> None:
    """node_text should extract a child node's source text."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    function_node = next(
        node
        for node in parser.walk(result.tree.root_node)
        if node.type == "function_definition"
    )

    text = parser.node_text(
        function_node,
        PYTHON_SOURCE,
    )

    assert text == PYTHON_SOURCE.rstrip()


def test_node_text_rejects_non_string_source() -> None:
    """node_text should reject non-string source values."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    with pytest.raises(
        ASTParserError,
        match="source must be a string",
    ):
        parser.node_text(
            result.tree.root_node,
            123,
        )


def test_node_metadata_contains_expected_fields() -> None:
    """node_metadata should return structural information."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    root = result.tree.root_node

    metadata = parser.node_metadata(root)

    assert metadata["type"] == "module"
    assert isinstance(metadata["named"], bool)
    assert isinstance(metadata["start_point"], dict)
    assert isinstance(metadata["end_point"], dict)
    assert isinstance(metadata["start_byte"], int)
    assert isinstance(metadata["end_byte"], int)
    assert isinstance(metadata["child_count"], int)


def test_node_metadata_contains_valid_positions() -> None:
    """Root-node metadata should contain valid source positions."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    metadata = parser.node_metadata(
        result.tree.root_node,
    )

    assert metadata["start_point"]["row"] == 0
    assert metadata["start_point"]["column"] == 0
    assert metadata["end_byte"] == len(
        PYTHON_SOURCE.encode("utf-8"),
    )


def test_parse_result_contains_tree_object() -> None:
    """ASTParseResult should expose the generated syntax tree."""
    parser = ASTParser()

    result = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    assert result.tree is not None
    assert result.tree.root_node is not None


def test_parser_is_deterministic() -> None:
    """Parsing identical input should produce identical summary data."""
    parser = ASTParser()

    first = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )
    second = parser.parse(
        PYTHON_SOURCE,
        ProgrammingLanguage.PYTHON,
    )

    assert first.root_node_type == second.root_node_type
    assert first.has_errors == second.has_errors
    assert first.error_node_count == second.error_node_count
    assert first.node_count == second.node_count