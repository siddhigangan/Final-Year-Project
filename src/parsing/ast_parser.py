"""AST parsing utilities for SecureCodeRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tree_sitter import Node, Parser, Tree

from src.models import ProgrammingLanguage
from src.parsing.language_registry import (
    LanguageRegistry,
    UnsupportedLanguageError,
)


class ASTParserError(RuntimeError):
    """Raised when source code cannot be parsed."""


class UnsupportedASTLanguageError(ASTParserError):
    """Raised when no AST parser exists for a language."""


@dataclass(frozen=True)
class ASTParseResult:
    """Result of parsing a source file."""

    language: ProgrammingLanguage
    tree: Tree
    root_node_type: str
    has_errors: bool
    error_node_count: int
    node_count: int


class ASTParser:
    """Parse supported programming languages using Tree-sitter."""

    def __init__(
        self,
        registry: LanguageRegistry | None = None,
    ) -> None:
        """Initialize the AST parser."""
        self.registry = registry or LanguageRegistry()
        self._parsers: dict[ProgrammingLanguage, Parser] = {}

    def parse(
        self,
        source: str,
        language: ProgrammingLanguage,
    ) -> ASTParseResult:
        """Parse source code into a Tree-sitter syntax tree."""
        if not isinstance(source, str):
            raise ASTParserError("source must be a string")

        if not isinstance(language, ProgrammingLanguage):
            raise ASTParserError(
                "language must be a ProgrammingLanguage value"
            )

        if not source:
            raise ASTParserError("source must not be empty")

        parser = self._get_parser(language)

        try:
            tree = parser.parse(source.encode("utf-8"))
        except Exception as exc:
            raise ASTParserError(
                f"Failed to parse {language.value} source"
            ) from exc

        root_node = tree.root_node
        error_node_count = self._count_error_nodes(root_node)
        node_count = self._count_nodes(root_node)

        return ASTParseResult(
            language=language,
            tree=tree,
            root_node_type=root_node.type,
            has_errors=error_node_count > 0,
            error_node_count=error_node_count,
            node_count=node_count,
        )

    def parse_bytes(
        self,
        source: bytes,
        language: ProgrammingLanguage,
    ) -> ASTParseResult:
        """Parse UTF-8 encoded source bytes."""
        if not isinstance(source, bytes):
            raise ASTParserError("source must be bytes")

        try:
            decoded_source = source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ASTParserError(
                "source bytes must contain valid UTF-8"
            ) from exc

        return self.parse(
            decoded_source,
            language,
        )

    def is_supported(
        self,
        language: ProgrammingLanguage,
    ) -> bool:
        """Return whether a language has an AST parser."""
        return self.registry.is_supported(language)

    def parser_for(
        self,
        language: ProgrammingLanguage,
    ) -> Parser:
        """Return the cached Tree-sitter parser for a language."""
        return self._get_parser(language)

    def clear_cache(self) -> None:
        """Clear cached parser instances."""
        self._parsers.clear()
        self.registry.clear_cache()

    def _get_parser(
        self,
        language: ProgrammingLanguage,
    ) -> Parser:
        """Create or retrieve a parser for a language."""
        if not isinstance(language, ProgrammingLanguage):
            raise ASTParserError(
                "language must be a ProgrammingLanguage value"
            )

        if language in self._parsers:
            return self._parsers[language]

        try:
            tree_sitter_language = self.registry.get(language)
        except UnsupportedLanguageError as exc:
            raise UnsupportedASTLanguageError(
                f"AST parsing is not supported for {language.value}"
            ) from exc

        try:
            parser = Parser(tree_sitter_language)
        except Exception as exc:
            raise ASTParserError(
                f"Failed to create parser for {language.value}"
            ) from exc

        self._parsers[language] = parser

        return parser

    @classmethod
    def _count_error_nodes(
        cls,
        node: Node,
    ) -> int:
        """Count syntax-error nodes recursively."""
        count = 1 if node.type == "ERROR" else 0

        for child in node.children:
            count += cls._count_error_nodes(child)

        return count

    @classmethod
    def _count_nodes(
        cls,
        node: Node,
    ) -> int:
        """Count all AST nodes recursively."""
        count = 1

        for child in node.children:
            count += cls._count_nodes(child)

        return count

    @staticmethod
    def walk(
        node: Node,
    ) -> list[Node]:
        """Return a depth-first list of AST nodes."""
        nodes: list[Node] = []
        stack: list[Node] = [node]

        while stack:
            current = stack.pop()
            nodes.append(current)
            stack.extend(reversed(current.children))

        return nodes

    @staticmethod
    def node_text(
        node: Node,
        source: str,
    ) -> str:
        """Return the source text represented by an AST node."""
        if not isinstance(source, str):
            raise ASTParserError("source must be a string")

        try:
            source_bytes = source.encode("utf-8")
            return source_bytes[node.start_byte:node.end_byte].decode(
                "utf-8"
            )
        except (UnicodeDecodeError, IndexError) as exc:
            raise ASTParserError(
                "Unable to extract text for AST node"
            ) from exc

    @staticmethod
    def node_metadata(
        node: Node,
    ) -> dict[str, Any]:
        """Return serializable structural metadata for an AST node."""
        return {
            "type": node.type,
            "named": node.is_named,
            "start_point": {
                "row": node.start_point.row,
                "column": node.start_point.column,
            },
            "end_point": {
                "row": node.end_point.row,
                "column": node.end_point.column,
            },
            "start_byte": node.start_byte,
            "end_byte": node.end_byte,
            "child_count": node.child_count,
        }