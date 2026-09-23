"""Tests for repository ingestion models."""

from __future__ import annotations

from src.ingestion.models import (
    RepositorySnapshot,
    SourceFile,
)


def make_source_file() -> SourceFile:
    """Create a deterministic test source file."""
    return SourceFile(
        file_id="file-001",
        repository_id="repo-001",
        relative_path="src/example.py",
        absolute_path="C:/repo/src/example.py",
        language="python",
        content="def hello():\n    return 'hello'\n",
        content_hash="abc123",
        size_bytes=32,
        line_count=2,
    )


class TestSourceFile:
    """Tests for SourceFile."""

    def test_file_name(self) -> None:
        source_file = make_source_file()

        assert source_file.file_name == "example.py"

    def test_extension(self) -> None:
        source_file = make_source_file()

        assert source_file.extension == ".py"

    def test_non_empty_file(self) -> None:
        source_file = make_source_file()

        assert source_file.is_empty is False

    def test_empty_file(self) -> None:
        source_file = SourceFile(
            file_id="file-002",
            repository_id="repo-001",
            relative_path="empty.py",
            absolute_path="C:/repo/empty.py",
            language="python",
            content="",
            content_hash="empty",
            size_bytes=0,
            line_count=0,
        )

        assert source_file.is_empty is True

    def test_to_dict(self) -> None:
        source_file = make_source_file()

        result = source_file.to_dict()

        assert result["file_id"] == "file-001"
        assert result["repository_id"] == "repo-001"
        assert result["relative_path"] == "src/example.py"
        assert result["language"] == "python"
        assert result["content_hash"] == "abc123"
        assert result["line_count"] == 2
        assert result["metadata"] == {}


class TestRepositorySnapshot:
    """Tests for RepositorySnapshot."""

    def test_successful_files(self) -> None:
        source_file = make_source_file()

        snapshot = RepositorySnapshot(
            repository_id="repo-001",
            root_path="C:/repo",
            files=(source_file,),
            total_files=3,
            skipped_files=1,
            failed_files=1,
        )

        assert snapshot.successful_files == 1

    def test_total_lines(self) -> None:
        first = make_source_file()

        second = SourceFile(
            file_id="file-002",
            repository_id="repo-001",
            relative_path="src/other.py",
            absolute_path="C:/repo/src/other.py",
            language="python",
            content="x = 1\n",
            content_hash="def456",
            size_bytes=6,
            line_count=1,
        )

        snapshot = RepositorySnapshot(
            repository_id="repo-001",
            root_path="C:/repo",
            files=(first, second),
            total_files=2,
            skipped_files=0,
            failed_files=0,
        )

        assert snapshot.total_lines == 3

    def test_to_dict(self) -> None:
        source_file = make_source_file()

        snapshot = RepositorySnapshot(
            repository_id="repo-001",
            root_path="C:/repo",
            files=(source_file,),
            total_files=1,
            skipped_files=0,
            failed_files=0,
        )

        result = snapshot.to_dict()

        assert result["repository_id"] == "repo-001"
        assert result["total_files"] == 1
        assert result["successful_files"] == 1
        assert result["total_lines"] == 2
        assert len(result["files"]) == 1