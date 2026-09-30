"""Unit tests for the SecureCodeRAG AST chunker."""

from __future__ import annotations

import pytest

from src.chunking.ast_chunker import (
    ASTChunker,
    ASTChunkerError,
    ChunkingConfig,
)
from src.chunking.chunk_models import ASTCodeChunk
from src.models import CodeChunk, ProgrammingLanguage, SourceTrust


@pytest.fixture
def chunker() -> ASTChunker:
    """Return a default AST chunker."""
    return ASTChunker()


class TestChunkingConfig:
    """Tests for chunking configuration."""

    def test_default_configuration(self) -> None:
        """Default configuration should contain safe values."""
        config = ChunkingConfig()

        assert config.max_lines == 80
        assert config.max_bytes == 8000
        assert config.min_lines == 1
        assert config.include_documentation is True
        assert config.include_nested_symbols is False
        assert config.fallback_to_top_level is True

    def test_zero_max_lines_rejected(self) -> None:
        """Maximum lines must be positive."""
        with pytest.raises(ASTChunkerError):
            ChunkingConfig(max_lines=0)

    def test_zero_max_bytes_rejected(self) -> None:
        """Maximum bytes must be positive."""
        with pytest.raises(ASTChunkerError):
            ChunkingConfig(max_bytes=0)

    def test_zero_min_lines_rejected(self) -> None:
        """Minimum lines must be positive."""
        with pytest.raises(ASTChunkerError):
            ChunkingConfig(min_lines=0)

    def test_min_lines_cannot_exceed_max_lines(self) -> None:
        """Minimum lines cannot exceed maximum lines."""
        with pytest.raises(ASTChunkerError):
            ChunkingConfig(
                min_lines=10,
                max_lines=5,
            )


class TestPythonChunking:
    """Tests for Python AST chunking."""

    def test_function_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A Python function should become a function chunk."""
        source = """def add(a, b):
    return a + b
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/math.py",
            trust=SourceTrust.TRUSTED,
        )

        assert result.language is ProgrammingLanguage.PYTHON
        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert isinstance(chunk, ASTCodeChunk)
        assert chunk.symbol_name == "add"
        assert chunk.metadata.symbol_type == "function"
        assert chunk.metadata.ast_node_type == "function_definition"
        assert chunk.trust is SourceTrust.TRUSTED
        assert "return a + b" in chunk.content

    def test_class_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A Python class should become a class chunk."""
        source = """class Calculator:
    def add(self, a, b):
        return a + b
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/calculator.py",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "Calculator"
        assert chunk.metadata.symbol_type == "class"
        assert chunk.metadata.ast_node_type == "class_definition"
        assert "def add" in chunk.content

    def test_multiple_top_level_functions(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Separate top-level functions should produce separate chunks."""
        source = """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/math.py",
        )

        assert result.chunk_count == 2

        names = [chunk.symbol_name for chunk in result.chunks]

        assert names == ["add", "subtract"]

    def test_parent_symbol_with_nested_symbols_enabled(
        self,
    ) -> None:
        """Nested symbols should retain their parent when enabled."""
        chunker = ASTChunker(
            config=ChunkingConfig(
                include_nested_symbols=True,
            )
        )

        source = """class UserService:
    def login(self, username):
        return username
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/user.py",
        )

        assert result.chunk_count == 2

        class_chunk = next(
            chunk
            for chunk in result.chunks
            if chunk.symbol_name == "UserService"
        )

        method_chunk = next(
            chunk
            for chunk in result.chunks
            if chunk.symbol_name == "login"
        )

        assert class_chunk.metadata.symbol_type == "class"
        assert method_chunk.metadata.symbol_type == "function"
        assert method_chunk.metadata.parent_symbol == "UserService"

    def test_nested_symbols_disabled_by_default(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Nested symbols should be excluded by default."""
        source = """class UserService:
    def login(self, username):
        return username
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/user.py",
        )

        assert result.chunk_count == 1
        assert result.chunks[0].symbol_name == "UserService"

    def test_chunk_indexes_are_deterministic(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Chunk indexes should be assigned in source order."""
        source = """def first():
    return 1

def second():
    return 2

def third():
    return 3
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/functions.py",
        )

        assert [chunk.metadata.chunk_index for chunk in result.chunks] == [
            0,
            1,
            2,
        ]

        assert all(
            chunk.metadata.total_chunks == 3
            for chunk in result.chunks
        )


class TestJavaScriptChunking:
    """Tests for JavaScript AST chunking."""

    def test_function_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A JavaScript function should be detected."""
        source = """function greet(name) {
    return "Hello " + name;
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.JAVASCRIPT,
            repository_id="repo-js",
            source_file_id="file-js",
            relative_path="src/greet.js",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "greet"
        assert chunk.metadata.symbol_type == "function"
        assert chunk.metadata.ast_node_type == "function_declaration"

    def test_class_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A JavaScript class should be detected."""
        source = """class User {
    constructor(name) {
        this.name = name;
    }
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.JAVASCRIPT,
            repository_id="repo-js",
            source_file_id="file-js",
            relative_path="src/user.js",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "User"
        assert chunk.metadata.symbol_type == "class"


class TestTypeScriptChunking:
    """Tests for TypeScript AST chunking."""

    def test_function_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A TypeScript function should be detected."""
        source = """function add(a: number, b: number): number {
    return a + b;
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.TYPESCRIPT,
            repository_id="repo-ts",
            source_file_id="file-ts",
            relative_path="src/math.ts",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "add"
        assert chunk.metadata.symbol_type == "function"
        assert chunk.metadata.ast_node_type == "function_declaration"


class TestJavaChunking:
    """Tests for Java AST chunking."""

    def test_class_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A Java class should be detected."""
        source = """class Calculator {
    int add(int a, int b) {
        return a + b;
    }
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.JAVA,
            repository_id="repo-java",
            source_file_id="file-java",
            relative_path="src/Calculator.java",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "Calculator"
        assert chunk.metadata.symbol_type == "class"
        assert chunk.metadata.ast_node_type == "class_declaration"

    def test_nested_java_method(
        self,
    ) -> None:
        """Java methods should be available when nested symbols are enabled."""
        chunker = ASTChunker(
            config=ChunkingConfig(
                include_nested_symbols=True,
            )
        )

        source = """class Calculator {
    int add(int a, int b) {
        return a + b;
    }
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.JAVA,
            repository_id="repo-java",
            source_file_id="file-java",
            relative_path="src/Calculator.java",
        )

        method = next(
            chunk
            for chunk in result.chunks
            if chunk.symbol_name == "add"
        )

        assert method.metadata.symbol_type == "method"
        assert method.metadata.parent_symbol == "Calculator"


class TestCAndCppChunking:
    """Tests for C and C++ AST chunking."""

    def test_c_function_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A C function should be detected."""
        source = """int add(int a, int b) {
    return a + b;
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.C,
            repository_id="repo-c",
            source_file_id="file-c",
            relative_path="src/math.c",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "add"
        assert chunk.metadata.symbol_type == "function"

    def test_cpp_function_is_chunked(
        self,
        chunker: ASTChunker,
    ) -> None:
        """A C++ function should be detected."""
        source = """int add(int a, int b) {
    return a + b;
}
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.CPP,
            repository_id="repo-cpp",
            source_file_id="file-cpp",
            relative_path="src/math.cpp",
        )

        assert result.chunk_count == 1

        chunk = result.chunks[0]

        assert chunk.symbol_name == "add"
        assert chunk.metadata.symbol_type == "function"


class TestFallbackBehavior:
    """Tests for non-symbol and oversized source handling."""

    def test_source_without_symbols_uses_fallback(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Source without recognized symbols should use root fallback."""
        source = """#include <stdio.h>

#define VALUE 42
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.C,
            repository_id="repo-c",
            source_file_id="file-c",
            relative_path="src/config.c",
        )

        assert result.chunk_count == 1
        assert result.chunks[0].metadata.ast_node_type == "translation_unit"
        assert "#define VALUE 42" in result.chunks[0].content

    def test_unsupported_chunking_language_uses_fallback(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Supported parser languages without symbol mappings use fallback."""
        source = "package main\n"

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.GO,
            repository_id="repo-go",
            source_file_id="file-go",
            relative_path="main.go",
        )

        assert result.chunk_count == 1
        assert result.chunks[0].content == source

    def test_oversized_source_is_split(
        self,
    ) -> None:
        """Oversized source should be bounded by configured limits."""
        chunker = ASTChunker(
            config=ChunkingConfig(
                max_lines=3,
                max_bytes=100,
            )
        )

        source = "\n".join(
            f"# comment {index}"
            for index in range(12)
        )

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-python",
            source_file_id="file-python",
            relative_path="src/comments.py",
        )

        assert result.chunk_count > 1

        assert all(
            chunk.line_count <= 3
            for chunk in result.chunks
        )


class TestMetadataAndSerialization:
    """Tests for metadata propagation."""

    def test_trust_is_preserved(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Source trust should propagate to every chunk."""
        source = """def secure():
    return True
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-secure",
            source_file_id="file-secure",
            relative_path="src/security.py",
            trust=SourceTrust.SUSPICIOUS,
        )

        assert all(
            chunk.trust is SourceTrust.SUSPICIOUS
            for chunk in result.chunks
        )

    def test_relative_path_is_preserved(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Relative source path should survive chunking."""
        result = chunker.chunk_source(
            source="def hello():\n    return 1\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-1",
            source_file_id="file-1",
            relative_path="nested/source.py",
        )

        assert result.chunks[0].metadata.relative_path == (
            "nested/source.py"
        )

    def test_to_code_chunks_returns_global_model(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Chunking result should convert to global CodeChunk objects."""
        result = chunker.chunk_source(
            source="def hello():\n    return 1\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-1",
            source_file_id="file-1",
            relative_path="source.py",
        )

        code_chunks = result.to_code_chunks()

        assert len(code_chunks) == 1
        assert isinstance(code_chunks[0], CodeChunk)

    def test_chunk_code_returns_global_models(
        self,
        chunker: ASTChunker,
    ) -> None:
        """chunk_code should directly return CodeChunk objects."""
        chunks = chunker.chunk_code(
            source="def hello():\n    return 1\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-1",
            source_file_id="file-1",
            relative_path="source.py",
        )

        assert len(chunks) == 1
        assert isinstance(chunks[0], CodeChunk)


class TestDocumentation:
    """Tests for documentation handling."""

    def test_documentation_is_included_by_default(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Documentation should be retained by default."""
        source = "# This is documentation\n"

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-docs",
            source_file_id="file-docs",
            relative_path="README.md",
            is_documentation=True,
        )

        assert result.chunk_count == 1
        assert result.chunks[0].metadata.is_documentation is True

    def test_documentation_can_be_excluded(self) -> None:
        """Documentation can be disabled through configuration."""
        chunker = ASTChunker(
            config=ChunkingConfig(
                include_documentation=False,
            )
        )

        result = chunker.chunk_source(
            source="# Documentation\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo-docs",
            source_file_id="file-docs",
            relative_path="README.md",
            is_documentation=True,
        )

        assert result.chunk_count == 0


class TestErrors:
    """Tests for invalid chunker inputs."""

    def test_empty_source_rejected(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Empty source should be rejected."""
        with pytest.raises(ASTChunkerError):
            chunker.chunk_source(
                source="",
                language=ProgrammingLanguage.PYTHON,
                repository_id="repo",
                source_file_id="file",
                relative_path="source.py",
            )

    def test_empty_repository_id_rejected(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Empty repository ID should be rejected."""
        with pytest.raises(ASTChunkerError):
            chunker.chunk_source(
                source="print('hello')",
                language=ProgrammingLanguage.PYTHON,
                repository_id="",
                source_file_id="file",
                relative_path="source.py",
            )

    def test_empty_source_file_id_rejected(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Empty source file ID should be rejected."""
        with pytest.raises(ASTChunkerError):
            chunker.chunk_source(
                source="print('hello')",
                language=ProgrammingLanguage.PYTHON,
                repository_id="repo",
                source_file_id="",
                relative_path="source.py",
            )

    def test_empty_relative_path_rejected(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Empty relative path should be rejected."""
        with pytest.raises(ASTChunkerError):
            chunker.chunk_source(
                source="print('hello')",
                language=ProgrammingLanguage.PYTHON,
                repository_id="repo",
                source_file_id="file",
                relative_path="",
            )

    def test_invalid_language_rejected(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Invalid language values should be rejected."""
        with pytest.raises(ASTChunkerError):
            chunker.chunk_source(
                source="print('hello')",
                language="python",  # type: ignore[arg-type]
                repository_id="repo",
                source_file_id="file",
                relative_path="source.py",
            )

    def test_invalid_trust_rejected(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Invalid trust values should be rejected."""
        with pytest.raises(ASTChunkerError):
            chunker.chunk_source(
                source="print('hello')",
                language=ProgrammingLanguage.PYTHON,
                repository_id="repo",
                source_file_id="file",
                relative_path="source.py",
                trust="trusted",  # type: ignore[arg-type]
            )


class TestDeterminism:
    """Tests for deterministic chunk generation."""

    def test_same_source_produces_same_chunk_ids(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Identical inputs should produce identical chunk IDs."""
        source = """def calculate(value):
    return value * 2
"""

        first = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo",
            source_file_id="file",
            relative_path="calculate.py",
        )

        second = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo",
            source_file_id="file",
            relative_path="calculate.py",
        )

        first_ids = [
            chunk.chunk_id
            for chunk in first.chunks
        ]
        second_ids = [
            chunk.chunk_id
            for chunk in second.chunks
        ]

        assert first_ids == second_ids

    def test_changed_source_changes_chunk_id(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Changing source content should change chunk identity."""
        first = chunker.chunk_source(
            source="def calculate():\n    return 1\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo",
            source_file_id="file",
            relative_path="calculate.py",
        )

        second = chunker.chunk_source(
            source="def calculate():\n    return 2\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo",
            source_file_id="file",
            relative_path="calculate.py",
        )

        assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


class TestParseStatus:
    """Tests for parser status propagation."""

    def test_valid_source_has_no_parse_errors(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Valid source should report no parser errors."""
        result = chunker.chunk_source(
            source="def hello():\n    return 1\n",
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo",
            source_file_id="file",
            relative_path="hello.py",
        )

        assert result.parse_had_errors is False
        assert result.source_node_count > 0

    def test_syntax_errors_are_recorded(
        self,
        chunker: ASTChunker,
    ) -> None:
        """Malformed source should preserve parser error information."""
        source = """def broken(
    return 42
"""

        result = chunker.chunk_source(
            source=source,
            language=ProgrammingLanguage.PYTHON,
            repository_id="repo",
            source_file_id="file",
            relative_path="broken.py",
        )

        assert result.parse_had_errors is True
        assert result.chunk_count >= 1