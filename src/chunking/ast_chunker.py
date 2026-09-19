"""AST-aware source-code chunking for SecureCodeRAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from tree_sitter import Node

from src.chunking.chunk_models import (
    ASTCodeChunk,
    ASTNodeInfo,
    ChunkMetadata,
    build_chunk_id,
)
from src.models import CodeChunk, ProgrammingLanguage, SourceTrust
from src.parsing.ast_parser import (
    ASTParser,
    ASTParseResult,
    UnsupportedASTLanguageError,
)


class ASTChunkerError(ValueError):
    """Raised when AST chunking fails."""


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration controlling AST-aware chunk generation."""

    max_lines: int = 80
    max_bytes: int = 8000
    min_lines: int = 1
    include_documentation: bool = True
    include_nested_symbols: bool = False
    fallback_to_top_level: bool = True

    def __post_init__(self) -> None:
        """Validate chunking configuration."""
        if self.max_lines < 1:
            raise ASTChunkerError("max_lines must be at least 1")

        if self.max_bytes < 1:
            raise ASTChunkerError("max_bytes must be at least 1")

        if self.min_lines < 1:
            raise ASTChunkerError("min_lines must be at least 1")

        if self.min_lines > self.max_lines:
            raise ASTChunkerError(
                "min_lines must not exceed max_lines"
            )


@dataclass(frozen=True)
class ChunkingResult:
    """Result of chunking a parsed source file."""

    language: ProgrammingLanguage
    chunks: tuple[ASTCodeChunk, ...]
    parse_had_errors: bool
    source_node_count: int

    @property
    def chunk_count(self) -> int:
        """Return the number of generated chunks."""
        return len(self.chunks)

    def to_code_chunks(self) -> list[CodeChunk]:
        """Convert all AST chunks to system-level CodeChunk objects."""
        return [chunk.to_code_chunk() for chunk in self.chunks]


class ASTChunker:
    """Generate retrieval-ready chunks from Tree-sitter ASTs."""

    _SYMBOL_TYPES: ClassVar[
        dict[ProgrammingLanguage, dict[str, str]]
    ] = {
        ProgrammingLanguage.PYTHON: {
            "function_definition": "function",
            "class_definition": "class",
        },
        ProgrammingLanguage.JAVASCRIPT: {
            "function_declaration": "function",
            "function": "function",
            "class_declaration": "class",
            "method_definition": "method",
        },
        ProgrammingLanguage.TYPESCRIPT: {
            "function_declaration": "function",
            "function": "function",
            "class_declaration": "class",
            "method_definition": "method",
        },
        ProgrammingLanguage.JAVA: {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        },
        ProgrammingLanguage.C: {
            "function_definition": "function",
            "struct_specifier": "struct",
        },
        ProgrammingLanguage.CPP: {
            "function_definition": "function",
            "class_specifier": "class",
            "struct_specifier": "struct",
        },
    }

    _NAME_FIELDS: ClassVar[tuple[str, ...]] = (
        "name",
        "declarator",
    )

    _IDENTIFIER_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "identifier",
            "type_identifier",
            "field_identifier",
            "property_identifier",
            "namespace_identifier",
        }
    )

    def __init__(
        self,
        parser: ASTParser | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        """Initialize the AST chunker."""
        self.parser = parser or ASTParser()
        self.config = config or ChunkingConfig()

    def chunk_source(
        self,
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust = SourceTrust.UNKNOWN,
        is_documentation: bool = False,
    ) -> ChunkingResult:
        """Parse and chunk a source file."""
        self._validate_source_inputs(
            source=source,
            language=language,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
        )

        if is_documentation and not self.config.include_documentation:
            return ChunkingResult(
                language=language,
                chunks=(),
                parse_had_errors=False,
                source_node_count=0,
            )

        try:
            parsed = self.parser.parse(source, language)
        except UnsupportedASTLanguageError:
            if self.config.fallback_to_top_level:
                return self._fallback_raw_source(
                    source=source,
                    language=language,
                    repository_id=repository_id,
                    source_file_id=source_file_id,
                    relative_path=relative_path,
                    trust=trust,
                    is_documentation=is_documentation,
                )

            raise
        except Exception as exc:
            raise ASTChunkerError(
                f"Failed to parse source for {language.value}"
            ) from exc

        return self.chunk_tree(
            parsed=parsed,
            source=source,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
            is_documentation=is_documentation,
        )

    def chunk_tree(
        self,
        parsed: ASTParseResult,
        source: str,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust = SourceTrust.UNKNOWN,
        is_documentation: bool = False,
    ) -> ChunkingResult:
        """Generate chunks from an existing parsed syntax tree."""
        self._validate_tree_inputs(
            parsed=parsed,
            source=source,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
        )

        if is_documentation and not self.config.include_documentation:
            return ChunkingResult(
                language=parsed.language,
                chunks=(),
                parse_had_errors=False,
                source_node_count=0,
            )

        if parsed.language not in self._SYMBOL_TYPES:
            return self._fallback_chunk(
                parsed=parsed,
                source=source,
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                trust=trust,
                is_documentation=is_documentation,
            )

        candidates = self._collect_candidates(
            parsed.tree.root_node,
            parsed.language,
        )

        if not candidates:
            return self._fallback_chunk(
                parsed=parsed,
                source=source,
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                trust=trust,
                is_documentation=is_documentation,
            )

        selected = self._select_candidates(candidates)

        chunks = self._build_chunks(
            nodes=selected,
            source=source,
            language=parsed.language,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
            is_documentation=is_documentation,
        )

        if not chunks and self.config.fallback_to_top_level:
            return self._fallback_chunk(
                parsed=parsed,
                source=source,
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                trust=trust,
                is_documentation=is_documentation,
            )

        finalized = self._finalize_chunks(chunks)

        return ChunkingResult(
            language=parsed.language,
            chunks=tuple(finalized),
            parse_had_errors=parsed.has_errors,
            source_node_count=parsed.node_count,
        )

    def chunk_code(
        self,
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust = SourceTrust.UNKNOWN,
        is_documentation: bool = False,
    ) -> list[CodeChunk]:
        """Return system-level CodeChunk objects."""
        result = self.chunk_source(
            source=source,
            language=language,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
            is_documentation=is_documentation,
        )

        return result.to_code_chunks()

    @staticmethod
    def _validate_source_inputs(
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
    ) -> None:
        """Validate inputs shared by source-based chunking."""
        if not source:
            raise ASTChunkerError("source must not be empty")

        if not repository_id.strip():
            raise ASTChunkerError("repository_id must not be empty")

        if not source_file_id.strip():
            raise ASTChunkerError("source_file_id must not be empty")

        if not relative_path.strip():
            raise ASTChunkerError("relative_path must not be empty")

        if not isinstance(language, ProgrammingLanguage):
            raise ASTChunkerError(
                "language must be a ProgrammingLanguage value"
            )

        if not isinstance(trust, SourceTrust):
            raise ASTChunkerError(
                "trust must be a SourceTrust value"
            )

    @staticmethod
    def _validate_tree_inputs(
        parsed: ASTParseResult,
        source: str,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
    ) -> None:
        """Validate inputs shared by tree-based chunking."""
        if not source:
            raise ASTChunkerError("source must not be empty")

        if not repository_id.strip():
            raise ASTChunkerError("repository_id must not be empty")

        if not source_file_id.strip():
            raise ASTChunkerError("source_file_id must not be empty")

        if not relative_path.strip():
            raise ASTChunkerError("relative_path must not be empty")

        if not isinstance(parsed, ASTParseResult):
            raise ASTChunkerError(
                "parsed must be an ASTParseResult value"
            )

        if not isinstance(trust, SourceTrust):
            raise ASTChunkerError(
                "trust must be a SourceTrust value"
            )

    def _collect_candidates(
        self,
        root: Node,
        language: ProgrammingLanguage,
    ) -> list[Node]:
        """Collect meaningful symbol nodes from an AST."""
        symbol_types = self._SYMBOL_TYPES.get(language, {})

        candidates: list[Node] = []
        stack: list[Node] = [root]

        while stack:
            node = stack.pop()

            if node.type in symbol_types:
                candidates.append(node)

            stack.extend(reversed(node.children))

        candidates.sort(
            key=lambda node: (
                node.start_byte,
                -(node.end_byte - node.start_byte),
            )
        )

        return candidates

    def _select_candidates(
        self,
        candidates: list[Node],
    ) -> list[Node]:
        """Select non-overlapping AST symbols."""
        if self.config.include_nested_symbols:
            return candidates

        selected: list[Node] = []

        for candidate in candidates:
            if any(
                self._contains_node(existing, candidate)
                for existing in selected
            ):
                continue

            selected.append(candidate)

        return selected

    @staticmethod
    def _contains_node(
        parent: Node,
        child: Node,
    ) -> bool:
        """Return whether one AST node fully contains another."""
        return (
            parent.start_byte <= child.start_byte
            and parent.end_byte >= child.end_byte
            and (
                parent.start_byte != child.start_byte
                or parent.end_byte != child.end_byte
            )
        )

    def _build_chunks(
        self,
        nodes: list[Node],
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> list[ASTCodeChunk]:
        """Build chunks from selected AST nodes."""
        chunks: list[ASTCodeChunk] = []

        for node in nodes:
            content = self._node_text(node, source)

            if not content.strip():
                continue

            if not self._within_limits(node):
                chunks.extend(
                    self._split_large_node(
                        node=node,
                        source=source,
                        language=language,
                        repository_id=repository_id,
                        source_file_id=source_file_id,
                        relative_path=relative_path,
                        trust=trust,
                        is_documentation=is_documentation,
                    )
                )
                continue

            symbol_type = self._symbol_type(language, node)
            symbol_name = self._symbol_name(node, source)
            parent_symbol = self._find_parent_symbol(
                node=node,
                source=source,
                language=language,
            )

            ast_info = self._build_ast_info(node)

            metadata = ChunkMetadata(
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                language=language,
                trust=trust,
                is_documentation=is_documentation,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                parent_symbol=parent_symbol,
                ast_node_type=node.type,
            )

            chunk_id = build_chunk_id(
                source_file_id=source_file_id,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                content=content,
            )

            chunks.append(
                ASTCodeChunk(
                    chunk_id=chunk_id,
                    content=content,
                    ast=ast_info,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_large_node(
        self,
        node: Node,
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> list[ASTCodeChunk]:
        """Split an oversized AST node using child nodes."""
        children = [
            child
            for child in node.named_children
            if child.end_byte > child.start_byte
        ]

        if not children:
            return self._line_based_fallback(
                node=node,
                source=source,
                language=language,
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                trust=trust,
                is_documentation=is_documentation,
            )

        chunks: list[ASTCodeChunk] = []

        for child in children:
            child_content = self._node_text(child, source)

            if not child_content.strip():
                continue

            if (
                child.end_byte - child.start_byte > self.config.max_bytes
                or self._node_line_count(child) > self.config.max_lines
            ):
                chunks.extend(
                    self._split_large_node(
                        node=child,
                        source=source,
                        language=language,
                        repository_id=repository_id,
                        source_file_id=source_file_id,
                        relative_path=relative_path,
                        trust=trust,
                        is_documentation=is_documentation,
                    )
                )
                continue

            symbol_name = self._symbol_name(node, source)
            symbol_type = self._symbol_type(language, node)

            metadata = ChunkMetadata(
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                language=language,
                trust=trust,
                is_documentation=is_documentation,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                parent_symbol=None,
                ast_node_type=child.type,
            )

            ast_info = self._build_ast_info(child)

            chunk_id = build_chunk_id(
                source_file_id,
                child.start_byte,
                child.end_byte,
                child_content,
            )

            chunks.append(
                ASTCodeChunk(
                    chunk_id=chunk_id,
                    content=child_content,
                    ast=ast_info,
                    metadata=metadata,
                )
            )

        return chunks

    def _line_based_fallback(
        self,
        node: Node,
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> list[ASTCodeChunk]:
        """Create bounded chunks when AST children cannot be split."""
        content = self._node_text(node, source)
        lines = content.splitlines(keepends=True)

        if not lines:
            return []

        chunks: list[ASTCodeChunk] = []
        current_lines: list[str] = []
        current_bytes = 0
        chunk_start_line = node.start_point.row + 1

        for offset, line in enumerate(lines):
            line_bytes = len(line.encode("utf-8"))

            if (
                current_lines
                and (
                    len(current_lines) >= self.config.max_lines
                    or current_bytes + line_bytes > self.config.max_bytes
                )
            ):
                chunks.append(
                    self._build_line_chunk(
                        lines=current_lines,
                        start_line=chunk_start_line,
                        language=language,
                        repository_id=repository_id,
                        source_file_id=source_file_id,
                        relative_path=relative_path,
                        trust=trust,
                        is_documentation=is_documentation,
                    )
                )
                current_lines = []
                current_bytes = 0
                chunk_start_line = (
                    node.start_point.row + offset + 1
                )

            current_lines.append(line)
            current_bytes += line_bytes

        if current_lines:
            chunks.append(
                self._build_line_chunk(
                    lines=current_lines,
                    start_line=chunk_start_line,
                    language=language,
                    repository_id=repository_id,
                    source_file_id=source_file_id,
                    relative_path=relative_path,
                    trust=trust,
                    is_documentation=is_documentation,
                )
            )

        return chunks

    def _build_line_chunk(
        self,
        lines: list[str],
        start_line: int,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> ASTCodeChunk:
        """Build a chunk from a bounded group of lines."""
        content = "".join(lines)
        end_line = start_line + len(lines) - 1
        start_byte = 0
        end_byte = len(content.encode("utf-8"))

        ast_info = ASTNodeInfo(
            node_type="line_chunk",
            start_byte=start_byte,
            end_byte=end_byte,
            start_line=start_line,
            end_line=end_line,
        )

        metadata = ChunkMetadata(
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            language=language,
            trust=trust,
            is_documentation=is_documentation,
            ast_node_type="line_chunk",
        )

        chunk_id = build_chunk_id(
            source_file_id,
            start_byte,
            end_byte,
            content,
        )

        return ASTCodeChunk(
            chunk_id=chunk_id,
            content=content,
            ast=ast_info,
            metadata=metadata,
        )

    def _fallback_chunk(
        self,
        parsed: ASTParseResult,
        source: str,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> ChunkingResult:
        """Create a bounded fallback chunk when no symbols exist."""
        root = parsed.tree.root_node

        if (
            root.end_byte - root.start_byte <= self.config.max_bytes
            and self._node_line_count(root) <= self.config.max_lines
        ):
            content = source
            ast_info = self._build_ast_info(root)

            metadata = ChunkMetadata(
                repository_id=repository_id,
                source_file_id=source_file_id,
                relative_path=relative_path,
                language=parsed.language,
                trust=trust,
                is_documentation=is_documentation,
                ast_node_type=root.type,
            )

            chunk_id = build_chunk_id(
                source_file_id,
                root.start_byte,
                root.end_byte,
                content,
            )

            chunk = ASTCodeChunk(
                chunk_id=chunk_id,
                content=content,
                ast=ast_info,
                metadata=metadata,
            )

            return ChunkingResult(
                language=parsed.language,
                chunks=(chunk,),
                parse_had_errors=parsed.has_errors,
                source_node_count=parsed.node_count,
            )

        chunks = self._line_based_fallback(
            node=root,
            source=source,
            language=parsed.language,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
            is_documentation=is_documentation,
        )

        return ChunkingResult(
            language=parsed.language,
            chunks=tuple(chunks),
            parse_had_errors=parsed.has_errors,
            source_node_count=parsed.node_count,
        )

    def _fallback_raw_source(
        self,
        source: str,
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> ChunkingResult:
        """Create fallback chunks when no AST grammar is available."""
        lines = source.splitlines(keepends=True)

        if not lines:
            return ChunkingResult(
                language=language,
                chunks=(),
                parse_had_errors=False,
                source_node_count=0,
            )

        chunks = self._build_raw_line_chunks(
            lines=lines,
            language=language,
            repository_id=repository_id,
            source_file_id=source_file_id,
            relative_path=relative_path,
            trust=trust,
            is_documentation=is_documentation,
        )

        return ChunkingResult(
            language=language,
            chunks=tuple(self._finalize_chunks(chunks)),
            parse_had_errors=False,
            source_node_count=0,
        )

    def _build_raw_line_chunks(
        self,
        lines: list[str],
        language: ProgrammingLanguage,
        repository_id: str,
        source_file_id: str,
        relative_path: str,
        trust: SourceTrust,
        is_documentation: bool,
    ) -> list[ASTCodeChunk]:
        """Build chunks directly from source lines without an AST."""
        chunks: list[ASTCodeChunk] = []
        current_lines: list[str] = []
        current_bytes = 0
        start_line = 1

        for offset, line in enumerate(lines):
            line_bytes = len(line.encode("utf-8"))

            if (
                current_lines
                and (
                    len(current_lines) >= self.config.max_lines
                    or current_bytes + line_bytes > self.config.max_bytes
                )
            ):
                chunks.append(
                    self._build_line_chunk(
                        lines=current_lines,
                        start_line=start_line,
                        language=language,
                        repository_id=repository_id,
                        source_file_id=source_file_id,
                        relative_path=relative_path,
                        trust=trust,
                        is_documentation=is_documentation,
                    )
                )

                current_lines = []
                current_bytes = 0
                start_line = offset + 1

            current_lines.append(line)
            current_bytes += line_bytes

        if current_lines:
            chunks.append(
                self._build_line_chunk(
                    lines=current_lines,
                    start_line=start_line,
                    language=language,
                    repository_id=repository_id,
                    source_file_id=source_file_id,
                    relative_path=relative_path,
                    trust=trust,
                    is_documentation=is_documentation,
                )
            )

        return chunks

    def _finalize_chunks(
        self,
        chunks: list[ASTCodeChunk],
    ) -> list[ASTCodeChunk]:
        """Add deterministic chunk indexes and totals."""
        ordered = sorted(
            chunks,
            key=lambda chunk: (
                chunk.ast.start_byte,
                chunk.ast.end_byte,
                chunk.chunk_id,
            ),
        )

        total = len(ordered)
        finalized: list[ASTCodeChunk] = []

        for index, chunk in enumerate(ordered):
            metadata = ChunkMetadata(
                repository_id=chunk.metadata.repository_id,
                source_file_id=chunk.metadata.source_file_id,
                relative_path=chunk.metadata.relative_path,
                language=chunk.metadata.language,
                trust=chunk.metadata.trust,
                is_documentation=chunk.metadata.is_documentation,
                symbol_name=chunk.metadata.symbol_name,
                symbol_type=chunk.metadata.symbol_type,
                parent_symbol=chunk.metadata.parent_symbol,
                ast_node_type=chunk.metadata.ast_node_type,
                chunk_index=index,
                total_chunks=total,
                extra=chunk.metadata.extra,
            )

            finalized.append(
                ASTCodeChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    ast=chunk.ast,
                    metadata=metadata,
                )
            )

        return finalized

    def _within_limits(self, node: Node) -> bool:
        """Return whether an AST node fits configured limits."""
        return (
            node.end_byte - node.start_byte <= self.config.max_bytes
            and self._node_line_count(node) <= self.config.max_lines
        )

    @staticmethod
    def _node_line_count(node: Node) -> int:
        """Return the source line count represented by a node."""
        return node.end_point.row - node.start_point.row + 1

    @staticmethod
    def _node_text(
        node: Node,
        source: str,
    ) -> str:
        """Extract source text represented by an AST node."""
        source_bytes = source.encode("utf-8")

        try:
            return source_bytes[
                node.start_byte:node.end_byte
            ].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ASTChunkerError(
                "Unable to decode AST node text as UTF-8"
            ) from exc

    def _symbol_type(
        self,
        language: ProgrammingLanguage,
        node: Node,
    ) -> str | None:
        """Return the normalized symbol type for an AST node."""
        return self._SYMBOL_TYPES.get(language, {}).get(node.type)

    def _symbol_name(
        self,
        node: Node,
        source: str,
    ) -> str | None:
        """Extract a clean symbol name from an AST node."""
        for field_name in self._NAME_FIELDS:
            field_node = node.child_by_field_name(field_name)

            if field_node is None:
                continue

            identifier = self._find_identifier(field_node, source)

            if identifier:
                return identifier

            text = self._node_text(field_node, source).strip()

            if text:
                return text

        for child in node.named_children:
            if child.type in self._IDENTIFIER_TYPES:
                text = self._node_text(child, source).strip()

                if text:
                    return text

        return None

    def _find_identifier(
        self,
        node: Node,
        source: str,
    ) -> str | None:
        """Recursively find the identifier inside a declarator."""
        if node.type in self._IDENTIFIER_TYPES:
            text = self._node_text(node, source).strip()

            if text:
                return text

        for child in node.named_children:
            identifier = self._find_identifier(child, source)

            if identifier:
                return identifier

        return None

    def _find_parent_symbol(
        self,
        node: Node,
        source: str,
        language: ProgrammingLanguage,
    ) -> str | None:
        """Find the nearest enclosing symbol."""
        current = node.parent

        while current is not None:
            if current.type in self._SYMBOL_TYPES.get(
                language,
                {},
            ):
                return self._symbol_name(current, source)

            current = current.parent

        return None

    def _build_ast_info(self, node: Node) -> ASTNodeInfo:
        """Convert a Tree-sitter node to ASTNodeInfo."""
        return ASTNodeInfo(
            node_type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            start_column=node.start_point.column,
            end_column=node.end_point.column,
            is_named=node.is_named,
        )


__all__ = [
    "ASTChunker",
    "ASTChunkerError",
    "ChunkingConfig",
    "ChunkingResult",
]