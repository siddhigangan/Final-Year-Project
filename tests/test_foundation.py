"""
Foundation tests for SecureCodeRAG.

These tests validate the local foundation layer without requiring
external services, model downloads, network access, or vector databases.
"""

from pathlib import Path

import pytest

from src import __version__, get_logger, load_config
from src.config import ConfigurationError
from src.models import (
    CodeChunk,
    ExperimentCondition,
    GenerationRequest,
    GenerationStatus,
    ProgrammingLanguage,
    SecurityDecision,
    SecuritySeverity,
    SourceFile,
    SourceTrust,
)


def test_package_version() -> None:
    """The package should expose the expected version."""
    assert __version__ == "0.1.0"


def test_default_configuration() -> None:
    """Default configuration should load with expected research defaults."""
    config = load_config()

    assert config.project_name == "SecureCodeRAG"
    assert config.version == "0.1.0"
    assert config.top_k == 5
    assert config.seed == 42
    assert config.embedding_provider == "huggingface"
    assert config.embedding_model == "microsoft/unixcoder-base"
    assert config.vectorstore_provider == "faiss"
    assert config.code_llm_provider == "ollama"
    assert config.code_llm == "deepseek-coder:6.7b"


def test_configuration_paths() -> None:
    """Configured project paths should resolve to Path objects."""
    config = load_config()

    paths = config.resolved_paths()

    assert isinstance(paths, dict)

    for path in paths.values():
        assert isinstance(path, Path)


def test_seed_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEED should override the configuration-file default."""
    monkeypatch.setenv("SEED", "123")

    config = load_config()

    assert config.seed == 123


def test_top_k_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOP_K should override the configured retrieval limit."""
    monkeypatch.setenv("TOP_K", "10")

    config = load_config()

    assert config.top_k == 10


def test_invalid_seed_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid seed values should raise a configuration error."""
    monkeypatch.setenv("SEED", "not-an-integer")

    with pytest.raises(ConfigurationError):
        load_config()


def test_invalid_top_k_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-positive TOP_K values should be rejected."""
    monkeypatch.setenv("TOP_K", "0")

    with pytest.raises(ConfigurationError):
        load_config()


def test_source_file_model() -> None:
    """SourceFile should preserve repository metadata."""
    source = SourceFile(
        file_id="file-001",
        repository_id="repo-001",
        relative_path="src/example.py",
        language=ProgrammingLanguage.PYTHON,
        content="def add(a, b):\n    return a + b\n",
        size_bytes=32,
        sha256="a" * 64,
        trust=SourceTrust.TRUSTED,
    )

    assert source.file_id == "file-001"
    assert source.repository_id == "repo-001"
    assert source.relative_path == "src/example.py"
    assert source.language is ProgrammingLanguage.PYTHON
    assert source.trust is SourceTrust.TRUSTED
    assert source.size_bytes == 32
    assert len(source.sha256) == 64


def test_code_chunk_model() -> None:
    """CodeChunk should calculate line count correctly."""
    chunk = CodeChunk(
        chunk_id="chunk-001",
        source_file_id="file-001",
        repository_id="repo-001",
        content="def add(a, b):\n    return a + b\n",
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=2,
        symbol_name="add",
        symbol_type="function",
    )

    assert chunk.chunk_id == "chunk-001"
    assert chunk.source_file_id == "file-001"
    assert chunk.repository_id == "repo-001"
    assert chunk.line_count == 2
    assert chunk.symbol_name == "add"
    assert chunk.symbol_type == "function"


def test_code_chunk_rejects_empty_id() -> None:
    """CodeChunk should reject an empty chunk identifier."""
    with pytest.raises(ValueError):
        CodeChunk(
            chunk_id="",
            source_file_id="file-001",
            repository_id="repo-001",
            content="print('hello')",
            language=ProgrammingLanguage.PYTHON,
            start_line=1,
            end_line=1,
        )


def test_generation_request_model() -> None:
    """GenerationRequest should retain its request contract."""
    request = GenerationRequest(
        request_id="request-001",
        task="code_generation",
        query="Create a Python function that adds two numbers.",
        context=[],
    )

    assert request.request_id == "request-001"
    assert request.task == "code_generation"
    assert "adds two numbers" in request.query
    assert request.context == []


def test_enum_values_are_available() -> None:
    """Core enums required by the research pipeline should be available."""
    assert ProgrammingLanguage.PYTHON.value == "python"

    assert SourceTrust.TRUSTED.value == "trusted"
    assert SourceTrust.UNKNOWN.value == "unknown"
    assert SourceTrust.SUSPICIOUS.value == "suspicious"
    assert SourceTrust.POISONED.value == "poisoned"

    assert GenerationStatus.SUCCESS.value == "success"
    assert SecuritySeverity.HIGH.value == "high"
    assert SecurityDecision.PASS.value == "pass"

    assert (
        ExperimentCondition.CLEAN_NO_DEFENSE.value
        == "A_clean_no_defense"
    )


def test_logger_initialization(tmp_path: Path) -> None:
    """Logger should initialize with a file handler."""
    log_file = tmp_path / "logs" / "foundation.log"

    logger = get_logger(
        name="SecureCodeRAG.foundation.test",
        log_file=str(log_file),
        level="INFO",
    )

    logger.info("Foundation test message")

    assert logger.name == "SecureCodeRAG.foundation.test"
    assert log_file.exists()
    assert "Foundation test message" in log_file.read_text(
        encoding="utf-8"
    )


def test_logger_without_file() -> None:
    """Logger should support console-only operation."""
    logger = get_logger(
        name="SecureCodeRAG.console.test",
        log_file=None,
        level="INFO",
    )

    logger.info("Console-only foundation test")

    assert logger.name == "SecureCodeRAG.console.test"