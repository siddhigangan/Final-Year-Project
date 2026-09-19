"""Unit tests for the SecureCodeRAG ingestion layer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.ingestion.file_filter import FileFilter, FileFilterConfig
from src.ingestion.language_detector import LanguageDetector
from src.ingestion.metadata import (
    MetadataExtractionError,
    MetadataExtractor,
)
from src.ingestion.repository_loader import (
    InvalidRepositoryError,
    RepositoryLoader,
    RepositoryLoaderError,
    RepositoryNotFoundError,
)
from src.models import ProgrammingLanguage, SourceTrust


@pytest.fixture
def repository_root(tmp_path: Path) -> Path:
    """Create a representative temporary repository."""
    src_dir = tmp_path / "src"
    docs_dir = tmp_path / "docs"
    tests_dir = tmp_path / "tests"

    src_dir.mkdir()
    docs_dir.mkdir()
    tests_dir.mkdir()

    (src_dir / "main.py").write_text(
        "def hello():\n    return 'hello'\n",
        encoding="utf-8",
    )

    (src_dir / "app.js").write_text(
        "function hello() { return 'hello'; }\n",
        encoding="utf-8",
    )

    (src_dir / "service.java").write_text(
        "class Service { }\n",
        encoding="utf-8",
    )

    (docs_dir / "README.md").write_text(
        "# SecureCodeRAG\n\nProject documentation.\n",
        encoding="utf-8",
    )

    (tests_dir / "test_main.py").write_text(
        "def test_hello():\n    assert True\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def loader() -> RepositoryLoader:
    """Return a repository loader."""
    return RepositoryLoader()


@pytest.fixture
def detector() -> LanguageDetector:
    """Return a language detector."""
    return LanguageDetector()


@pytest.fixture
def metadata_extractor() -> MetadataExtractor:
    """Return a metadata extractor."""
    return MetadataExtractor()


def test_repository_loader_rejects_missing_repository(
    loader: RepositoryLoader,
    tmp_path: Path,
) -> None:
    """A missing repository path must be rejected."""
    missing_path = tmp_path / "does-not-exist"

    with pytest.raises(RepositoryNotFoundError):
        loader.load(missing_path)


def test_repository_loader_rejects_file_as_repository(
    loader: RepositoryLoader,
    tmp_path: Path,
) -> None:
    """A file cannot be used as a repository root."""
    repository_file = tmp_path / "repository.txt"

    repository_file.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(InvalidRepositoryError):
        loader.load(repository_file)


def test_repository_loader_discovers_files_deterministically(
    loader: RepositoryLoader,
    repository_root: Path,
) -> None:
    """Repository file discovery must be deterministic."""
    files_first = loader.discover_files(repository_root)
    files_second = loader.discover_files(repository_root)

    relative_first = [
        path.relative_to(repository_root).as_posix()
        for path in files_first
    ]

    relative_second = [
        path.relative_to(repository_root).as_posix()
        for path in files_second
    ]

    assert relative_first == relative_second
    assert relative_first == sorted(relative_first)


def test_repository_loader_excludes_generated_directories(
    loader: RepositoryLoader,
    repository_root: Path,
) -> None:
    """Generated and dependency directories must not be discovered."""
    node_modules = repository_root / "node_modules"
    git_dir = repository_root / ".git"
    pycache = repository_root / "__pycache__"

    node_modules.mkdir()
    git_dir.mkdir()
    pycache.mkdir()

    (node_modules / "package.js").write_text(
        "module.exports = {};",
        encoding="utf-8",
    )

    (git_dir / "config").write_text(
        "repository metadata",
        encoding="utf-8",
    )

    (pycache / "cached.pyc").write_bytes(b"binary")

    files = loader.discover_files(repository_root)

    relative_paths = {
        path.relative_to(repository_root).as_posix()
        for path in files
    }

    assert "node_modules/package.js" not in relative_paths
    assert ".git/config" not in relative_paths
    assert "__pycache__/cached.pyc" not in relative_paths


def test_repository_loader_rejects_path_outside_boundary(
    loader: RepositoryLoader,
    repository_root: Path,
    tmp_path: Path,
) -> None:
    """Repository loading must not ingest files outside its root."""
    outside_file = tmp_path.parent / "outside-securecoderag.py"

    outside_file.write_text(
        "print('outside')",
        encoding="utf-8",
    )

    with pytest.raises(RepositoryLoaderError):
        loader.load(outside_file)


def test_file_filter_accepts_supported_source_file(
    repository_root: Path,
) -> None:
    """Supported source files should be accepted."""
    source_file = repository_root / "src" / "main.py"
    file_filter = FileFilter()

    result = file_filter.evaluate(source_file)

    assert result.accepted is True
    assert result.reason


def test_file_filter_accepts_documentation(
    repository_root: Path,
) -> None:
    """Supported documentation files should be accepted."""
    documentation = repository_root / "docs" / "README.md"
    file_filter = FileFilter()

    result = file_filter.evaluate(documentation)

    assert result.accepted is True


def test_file_filter_rejects_binary_files(
    repository_root: Path,
) -> None:
    """Binary files must not enter the ingestion pipeline."""
    binary_file = repository_root / "image.png"

    binary_file.write_bytes(
        b"\x89PNG\r\n\x1a\n",
    )

    file_filter = FileFilter()

    result = file_filter.evaluate(binary_file)

    assert result.accepted is False


def test_file_filter_rejects_oversized_files(
    repository_root: Path,
) -> None:
    """Files exceeding the configured size must be rejected."""
    large_file = repository_root / "large.py"

    large_file.write_text(
        "x" * 100,
        encoding="utf-8",
    )

    config = FileFilterConfig(
        max_file_size_bytes=50,
    )

    file_filter = FileFilter(config)

    result = file_filter.evaluate(large_file)

    assert result.accepted is False
    assert "size" in result.reason.lower()


def test_file_filter_rejects_secret_like_files(
    repository_root: Path,
) -> None:
    """Secret-like files must be rejected."""
    secret_file = repository_root / ".env"

    secret_file.write_text(
        "OPENAI_API_KEY=example-secret\n",
        encoding="utf-8",
    )

    file_filter = FileFilter()

    result = file_filter.evaluate(secret_file)

    assert result.accepted is False


def test_language_detector_detects_supported_extensions(
    detector: LanguageDetector,
) -> None:
    """Supported source extensions must map to the correct language."""
    cases = {
        "main.py": ProgrammingLanguage.PYTHON,
        "app.js": ProgrammingLanguage.JAVASCRIPT,
        "component.tsx": ProgrammingLanguage.TYPESCRIPT,
        "Service.java": ProgrammingLanguage.JAVA,
        "engine.cpp": ProgrammingLanguage.CPP,
        "program.c": ProgrammingLanguage.C,
        "Program.cs": ProgrammingLanguage.CSHARP,
        "server.go": ProgrammingLanguage.GO,
        "main.rs": ProgrammingLanguage.RUST,
        "index.php": ProgrammingLanguage.PHP,
        "script.rb": ProgrammingLanguage.RUBY,
    }

    for filename, expected_language in cases.items():
        result = detector.detect(Path(filename))

        assert result.language is expected_language
        assert result.confidence == 1.0
        assert result.method == "extension"


def test_language_detector_returns_unknown_for_unsupported_extension(
    detector: LanguageDetector,
) -> None:
    """Unsupported extensions must resolve to UNKNOWN."""
    result = detector.detect(Path("data.scala"))

    assert result.language is ProgrammingLanguage.UNKNOWN
    assert result.confidence == 0.0
    assert result.method == "unknown"


def test_language_detector_handles_special_filenames(
    detector: LanguageDetector,
) -> None:
    """Special repository files should be classified as unknown."""
    for filename in (
        "Dockerfile",
        "Makefile",
        "README.md",
        "LICENSE",
    ):
        result = detector.detect(Path(filename))

        assert result.language is ProgrammingLanguage.UNKNOWN
        assert result.confidence == 1.0
        assert result.method == "filename"


def test_language_detector_detects_python_shebang(
    detector: LanguageDetector,
) -> None:
    """Python shebangs should be detected."""
    content = "#!/usr/bin/env python3\nprint('hello')\n"

    result = detector.detect(
        Path("script"),
        content,
    )

    assert result.language is ProgrammingLanguage.PYTHON
    assert result.confidence == 0.95
    assert result.method == "shebang"


def test_language_detector_detects_node_shebang(
    detector: LanguageDetector,
) -> None:
    """Node shebangs should map to JavaScript."""
    content = "#!/usr/bin/env node\nconsole.log('hello');\n"

    result = detector.detect(
        Path("script"),
        content,
    )

    assert result.language is ProgrammingLanguage.JAVASCRIPT
    assert result.confidence == 0.95
    assert result.method == "shebang"


def test_language_detector_detect_all_is_deterministic(
    detector: LanguageDetector,
) -> None:
    """Multiple-file detection must be deterministic."""
    paths = [
        Path("z.py"),
        Path("a.js"),
        Path("m.java"),
    ]

    results = detector.detect_all(paths)

    assert [result.path.name for result in results] == [
        "a.js",
        "m.java",
        "z.py",
    ]


def test_metadata_extractor_creates_source_file(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """Metadata extraction should create a valid SourceFile."""
    source_path = repository_root / "src" / "main.py"

    result = metadata_extractor.extract(
        "repo-test",
        repository_root,
        source_path,
    )

    assert result.repository_id == "repo-test"
    assert result.relative_path == "src/main.py"
    assert result.language is ProgrammingLanguage.PYTHON
    assert result.trust is SourceTrust.TRUSTED
    assert result.is_documentation is False
    assert result.content == "def hello():\n    return 'hello'\n"


def test_metadata_extractor_calculates_sha256(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """Metadata extraction should calculate content SHA-256."""
    source_path = repository_root / "src" / "main.py"
    content = source_path.read_text(encoding="utf-8")

    result = metadata_extractor.extract(
        "repo-test",
        repository_root,
        source_path,
    )

    expected_hash = hashlib.sha256(
        content.encode("utf-8"),
    ).hexdigest()

    assert result.sha256 == expected_hash
    assert len(result.sha256) == 64


def test_metadata_extractor_identifies_documentation(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """Documentation files must be marked as documentation."""
    documentation = repository_root / "docs" / "README.md"

    result = metadata_extractor.extract(
        "repo-test",
        repository_root,
        documentation,
    )

    assert result.is_documentation is True
    assert result.metadata["file_type"] == "documentation"


def test_metadata_extractor_identifies_source_code(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """Supported programming files must be marked as source code."""
    source_path = repository_root / "src" / "service.java"

    result = metadata_extractor.extract(
        "repo-test",
        repository_root,
        source_path,
    )

    assert result.is_documentation is False
    assert result.metadata["file_type"] == "source_code"
    assert result.language is ProgrammingLanguage.JAVA


def test_metadata_extractor_preserves_default_trust(
    repository_root: Path,
) -> None:
    """Metadata extraction should use trusted sources by default."""
    source_path = repository_root / "src" / "main.py"

    extractor = MetadataExtractor()

    result = extractor.extract(
        "repo-test",
        repository_root,
        source_path,
    )

    assert result.trust is SourceTrust.TRUSTED


def test_metadata_extractor_rejects_external_file(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
    tmp_path: Path,
) -> None:
    """Metadata extraction must enforce repository boundaries."""
    outside_file = tmp_path.parent / "outside-metadata.py"

    outside_file.write_text(
        "print('outside')",
        encoding="utf-8",
    )

    with pytest.raises(MetadataExtractionError):
        metadata_extractor.extract(
            "repo-test",
            repository_root,
            outside_file,
        )


def test_metadata_extractor_rejects_missing_file(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """Metadata extraction must reject missing files."""
    missing_file = repository_root / "src" / "missing.py"

    with pytest.raises(MetadataExtractionError):
        metadata_extractor.extract(
            "repo-test",
            repository_root,
            missing_file,
        )


def test_metadata_extractor_rejects_empty_repository_id(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """An empty repository identifier must be rejected."""
    source_path = repository_root / "src" / "main.py"

    with pytest.raises(MetadataExtractionError):
        metadata_extractor.extract(
            "",
            repository_root,
            source_path,
        )


def test_metadata_extractor_extract_all_is_deterministic(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """Bulk metadata extraction must be deterministic."""
    paths = [
        repository_root / "src" / "service.java",
        repository_root / "src" / "main.py",
        repository_root / "docs" / "README.md",
        repository_root / "src" / "app.js",
    ]

    results = metadata_extractor.extract_all(
        "repo-test",
        repository_root,
        paths,
    )

    relative_paths = [
        result.relative_path
        for result in results
    ]

    assert relative_paths == sorted(relative_paths)


def test_metadata_extractor_produces_stable_file_id(
    metadata_extractor: MetadataExtractor,
    repository_root: Path,
) -> None:
    """The same file must produce the same deterministic file ID."""
    source_path = repository_root / "src" / "main.py"

    first = metadata_extractor.extract(
        "repo-test",
        repository_root,
        source_path,
    )

    second = metadata_extractor.extract(
        "repo-test",
        repository_root,
        source_path,
    )

    assert first.file_id == second.file_id
    assert first.sha256 == second.sha256


def test_phase_one_components_import_together() -> None:
    """All Phase 1 ingestion components must be importable."""
    assert RepositoryLoader is not None
    assert FileFilter is not None
    assert LanguageDetector is not None
    assert MetadataExtractor is not None