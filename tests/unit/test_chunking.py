"""Unit tests for AST-aware chunking data models."""

from __future__ import annotations

import pytest

from src.chunking.chunk_models import (
    ASTCodeChunk,
    ASTNodeInfo,
    ChunkMetadata,
    ChunkModelError,
    build_chunk_id,
)
from src.models import CodeChunk, ProgrammingLanguage, SourceTrust


def make_ast() -> ASTNodeInfo:
    """Create a valid AST node fixture."""
    return ASTNodeInfo(
        node_type="function_definition",
        start_byte=0,
        end_byte=35,
        start_line=1,
        end_line=3,
        start_column=0,
        end_column=14,
    )


def make_metadata() -> ChunkMetadata:
    """Create valid chunk metadata."""
    return ChunkMetadata(
        repository_id="repo-123",
        source_file_id="file-123",
        relative_path="src/main.py",
        language=ProgrammingLanguage.PYTHON,
        trust=SourceTrust.TRUSTED,
        symbol_name="hello",
        symbol_type="function",
        parent_symbol=None,
        ast_node_type="function_definition",
        chunk_index=0,
        total_chunks=1,
        extra={"test": True},
    )


def make_chunk() -> ASTCodeChunk:
    """Create a valid AST code chunk."""
    content = "def hello():\n    return 42\n"

    return ASTCodeChunk(
        chunk_id=build_chunk_id(
            "file-123",
            0,
            len(content.encode("utf-8")),
            content,
        ),
        content=content,
        ast=make_ast(),
        metadata=make_metadata(),
    )


class TestASTNodeInfo:
    """Tests for ASTNodeInfo."""

    def test_valid_node(self) -> None:
        """A valid AST node should be accepted."""
        node = make_ast()

        assert node.node_type == "function_definition"
        assert node.start_byte == 0
        assert node.end_byte == 35
        assert node.start_line == 1
        assert node.end_line == 3
        assert node.start_column == 0
        assert node.end_column == 14
        assert node.is_named is True

    def test_byte_length(self) -> None:
        """Byte length should be derived from byte boundaries."""
        node = make_ast()

        assert node.byte_length == 35

    def test_line_count(self) -> None:
        """Line count should include both boundary lines."""
        node = make_ast()

        assert node.line_count == 3

    def test_empty_node_type_rejected(self) -> None:
        """An empty node type should be rejected."""
        with pytest.raises(ChunkModelError):
            ASTNodeInfo(
                node_type="",
                start_byte=0,
                end_byte=1,
                start_line=1,
                end_line=1,
            )

    def test_negative_start_byte_rejected(self) -> None:
        """Negative start byte should be rejected."""
        with pytest.raises(ChunkModelError):
            ASTNodeInfo(
                node_type="function_definition",
                start_byte=-1,
                end_byte=1,
                start_line=1,
                end_line=1,
            )

    def test_invalid_byte_range_rejected(self) -> None:
        """End byte before start byte should be rejected."""
        with pytest.raises(ChunkModelError):
            ASTNodeInfo(
                node_type="function_definition",
                start_byte=10,
                end_byte=5,
                start_line=1,
                end_line=1,
            )

    def test_invalid_start_line_rejected(self) -> None:
        """Source lines must start at one or later."""
        with pytest.raises(ChunkModelError):
            ASTNodeInfo(
                node_type="function_definition",
                start_byte=0,
                end_byte=1,
                start_line=0,
                end_line=1,
            )

    def test_invalid_line_range_rejected(self) -> None:
        """End line cannot precede start line."""
        with pytest.raises(ChunkModelError):
            ASTNodeInfo(
                node_type="function_definition",
                start_byte=0,
                end_byte=1,
                start_line=5,
                end_line=2,
            )

    def test_negative_columns_rejected(self) -> None:
        """Column positions must be non-negative."""
        with pytest.raises(ChunkModelError):
            ASTNodeInfo(
                node_type="function_definition",
                start_byte=0,
                end_byte=1,
                start_line=1,
                end_line=1,
                start_column=-1,
            )

    def test_to_dict(self) -> None:
        """AST node metadata should serialize correctly."""
        result = make_ast().to_dict()

        assert result["node_type"] == "function_definition"
        assert result["start_byte"] == 0
        assert result["end_byte"] == 35
        assert result["start_line"] == 1
        assert result["end_line"] == 3
        assert result["byte_length"] == 35
        assert result["line_count"] == 3


class TestChunkMetadata:
    """Tests for ChunkMetadata."""

    def test_valid_metadata(self) -> None:
        """Valid metadata should be accepted."""
        metadata = make_metadata()

        assert metadata.repository_id == "repo-123"
        assert metadata.source_file_id == "file-123"
        assert metadata.relative_path == "src/main.py"
        assert metadata.language is ProgrammingLanguage.PYTHON
        assert metadata.trust is SourceTrust.TRUSTED
        assert metadata.symbol_name == "hello"
        assert metadata.symbol_type == "function"
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 1

    def test_empty_repository_id_rejected(self) -> None:
        """Repository ID cannot be empty."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="",
                source_file_id="file-123",
                relative_path="src/main.py",
                language=ProgrammingLanguage.PYTHON,
            )

    def test_empty_source_file_id_rejected(self) -> None:
        """Source file ID cannot be empty."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="",
                relative_path="src/main.py",
                language=ProgrammingLanguage.PYTHON,
            )

    def test_empty_relative_path_rejected(self) -> None:
        """Relative path cannot be empty."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="file-123",
                relative_path="",
                language=ProgrammingLanguage.PYTHON,
            )

    def test_invalid_language_rejected(self) -> None:
        """Language must be a ProgrammingLanguage enum."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="file-123",
                relative_path="src/main.py",
                language="python",  # type: ignore[arg-type]
            )

    def test_invalid_trust_rejected(self) -> None:
        """Trust must be a SourceTrust enum."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="file-123",
                relative_path="src/main.py",
                language=ProgrammingLanguage.PYTHON,
                trust="trusted",  # type: ignore[arg-type]
            )

    def test_negative_chunk_index_rejected(self) -> None:
        """Chunk index cannot be negative."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="file-123",
                relative_path="src/main.py",
                language=ProgrammingLanguage.PYTHON,
                chunk_index=-1,
            )

    def test_invalid_total_chunks_rejected(self) -> None:
        """Total chunks must be at least one."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="file-123",
                relative_path="src/main.py",
                language=ProgrammingLanguage.PYTHON,
                total_chunks=0,
            )

    def test_chunk_index_must_fit_total(self) -> None:
        """Chunk index must be smaller than total chunks."""
        with pytest.raises(ChunkModelError):
            ChunkMetadata(
                repository_id="repo-123",
                source_file_id="file-123",
                relative_path="src/main.py",
                language=ProgrammingLanguage.PYTHON,
                chunk_index=2,
                total_chunks=2,
            )

    def test_to_dict(self) -> None:
        """Metadata should serialize enum values as strings."""
        result = make_metadata().to_dict()

        assert result["repository_id"] == "repo-123"
        assert result["source_file_id"] == "file-123"
        assert result["language"] == "python"
        assert result["trust"] == "trusted"
        assert result["symbol_name"] == "hello"
        assert result["extra"] == {"test": True}


class TestASTCodeChunk:
    """Tests for ASTCodeChunk."""

    def test_valid_chunk(self) -> None:
        """A valid chunk should be accepted."""
        chunk = make_chunk()

        assert chunk.content == "def hello():\n    return 42\n"
        assert chunk.language is ProgrammingLanguage.PYTHON
        assert chunk.start_line == 1
        assert chunk.end_line == 3
        assert chunk.line_count == 3
        assert chunk.symbol_name == "hello"
        assert chunk.trust is SourceTrust.TRUSTED

    def test_empty_chunk_id_rejected(self) -> None:
        """Chunk ID cannot be empty."""
        with pytest.raises(ChunkModelError):
            ASTCodeChunk(
                chunk_id="",
                content="print('hello')",
                ast=make_ast(),
                metadata=make_metadata(),
            )

    def test_empty_content_rejected(self) -> None:
        """Chunk content cannot be empty or whitespace."""
        with pytest.raises(ChunkModelError):
            ASTCodeChunk(
                chunk_id="chunk-123",
                content="   ",
                ast=make_ast(),
                metadata=make_metadata(),
            )

    def test_invalid_ast_rejected(self) -> None:
        """AST information must use ASTNodeInfo."""
        with pytest.raises(ChunkModelError):
            ASTCodeChunk(
                chunk_id="chunk-123",
                content="print('hello')",
                ast="invalid",  # type: ignore[arg-type]
                metadata=make_metadata(),
            )

    def test_invalid_metadata_rejected(self) -> None:
        """Metadata must use ChunkMetadata."""
        with pytest.raises(ChunkModelError):
            ASTCodeChunk(
                chunk_id="chunk-123",
                content="print('hello')",
                ast=make_ast(),
                metadata="invalid",  # type: ignore[arg-type]
            )

    def test_to_code_chunk(self) -> None:
        """ASTCodeChunk should convert to the global CodeChunk model."""
        chunk = make_chunk()

        result = chunk.to_code_chunk()

        assert isinstance(result, CodeChunk)
        assert result.chunk_id == chunk.chunk_id
        assert result.source_file_id == "file-123"
        assert result.repository_id == "repo-123"
        assert result.language is ProgrammingLanguage.PYTHON
        assert result.start_line == 1
        assert result.end_line == 3
        assert result.symbol_name == "hello"
        assert result.symbol_type == "function"
        assert result.ast_node_type == "function_definition"
        assert result.parent_symbol is None
        assert result.trust is SourceTrust.TRUSTED
        assert result.metadata["relative_path"] == "src/main.py"
        assert result.metadata["ast"]["node_type"] == "function_definition"

    def test_to_dict(self) -> None:
        """Chunk should serialize its content and metadata."""
        result = make_chunk().to_dict()

        assert "chunk_id" in result
        assert "content" in result
        assert "ast" in result
        assert "metadata" in result
        assert result["metadata"]["language"] == "python"


class TestChunkID:
    """Tests for deterministic chunk identifiers."""

    def test_deterministic(self) -> None:
        """Identical inputs should produce identical IDs."""
        first = build_chunk_id(
            "file-123",
            0,
            10,
            "print(42)",
        )
        second = build_chunk_id(
            "file-123",
            0,
            10,
            "print(42)",
        )

        assert first == second

    def test_changes_with_source_file(self) -> None:
        """Changing the source file should change the ID."""
        first = build_chunk_id(
            "file-123",
            0,
            10,
            "print(42)",
        )
        second = build_chunk_id(
            "file-456",
            0,
            10,
            "print(42)",
        )

        assert first != second

    def test_changes_with_byte_range(self) -> None:
        """Changing the byte range should change the ID."""
        first = build_chunk_id(
            "file-123",
            0,
            10,
            "print(42)",
        )
        second = build_chunk_id(
            "file-123",
            1,
            10,
            "print(42)",
        )

        assert first != second

    def test_changes_with_content(self) -> None:
        """Changing content should change the ID."""
        first = build_chunk_id(
            "file-123",
            0,
            10,
            "print(42)",
        )
        second = build_chunk_id(
            "file-123",
            0,
            10,
            "print(43)",
        )

        assert first != second

    def test_has_expected_prefix(self) -> None:
        """Chunk IDs should use the documented prefix."""
        chunk_id = build_chunk_id(
            "file-123",
            0,
            10,
            "print(42)",
        )

        assert chunk_id.startswith("chunk-")

    def test_empty_source_file_id_rejected(self) -> None:
        """Source file ID must not be empty."""
        with pytest.raises(ChunkModelError):
            build_chunk_id(
                "",
                0,
                10,
                "print(42)",
            )

    def test_negative_start_byte_rejected(self) -> None:
        """Start byte cannot be negative."""
        with pytest.raises(ChunkModelError):
            build_chunk_id(
                "file-123",
                -1,
                10,
                "print(42)",
            )

    def test_invalid_range_rejected(self) -> None:
        """End byte cannot precede start byte."""
        with pytest.raises(ChunkModelError):
            build_chunk_id(
                "file-123",
                10,
                5,
                "print(42)",
            )

    def test_empty_content_rejected(self) -> None:
        """Chunk content must not be empty."""
        with pytest.raises(ChunkModelError):
            build_chunk_id(
                "file-123",
                0,
                10,
                "",
            )