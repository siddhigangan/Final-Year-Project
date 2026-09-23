"""Tests for repository ingestion orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.repository_ingestor import (
    RepositoryIngestionConfig,
    RepositoryIngestor,
    SourceFileReadError,
)


def create_repository(
    root: Path,
) -> None:
    """Create a small deterministic test repository."""
    (root / "src").mkdir()

    (root / "src" / "hello.py").write_text(
        "def hello():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    (root / "src" / "app.js").write_text(
        "function hello() {\n"
        "    return 'hello';\n"
        "}\n",
        encoding="utf-8",
    )

    (root / "README.md").write_text(
        "# Example Repository\n",
        encoding="utf-8",
    )

    (root / "image.png").write_bytes(
        b"\x89PNG\r\n"
    )


class TestRepositoryIngestor:
    """Tests for RepositoryIngestor."""

    def test_ingests_supported_files(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        paths = {
            source_file.relative_path
            for source_file in snapshot.files
        }

        assert "src/hello.py" in paths
        assert "src/app.js" in paths
        assert "README.md" in paths

    def test_excludes_unsupported_files(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        paths = {
            source_file.relative_path
            for source_file in snapshot.files
        }

        assert "image.png" not in paths

    def test_total_file_count_is_preserved(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        assert snapshot.total_files == 4

    def test_skipped_files_are_counted(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        assert snapshot.skipped_files == 1

    def test_python_language_is_detected(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        source_file = next(
            item
            for item in snapshot.files
            if item.relative_path == "src/hello.py"
        )

        assert source_file.language == "python"

    def test_javascript_language_is_detected(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        source_file = next(
            item
            for item in snapshot.files
            if item.relative_path == "src/app.js"
        )

        assert source_file.language == "javascript"

    def test_content_is_preserved(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        source_file = next(
            item
            for item in snapshot.files
            if item.relative_path == "src/hello.py"
        )

        assert "return 'hello'" in source_file.content

    def test_include_content_can_be_disabled(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        config = RepositoryIngestionConfig(
            include_content=False,
        )

        ingestor = RepositoryIngestor(
            config=config,
        )

        snapshot = ingestor.ingest(tmp_path)

        source_file = next(
            item
            for item in snapshot.files
            if item.relative_path == "src/hello.py"
        )

        assert source_file.content == ""

    def test_line_count_is_calculated(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        source_file = next(
            item
            for item in snapshot.files
            if item.relative_path == "src/hello.py"
        )

        assert source_file.line_count == 2

    def test_content_hash_is_deterministic(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        first = ingestor.ingest(tmp_path)
        second = ingestor.ingest(tmp_path)

        first_file = next(
            item
            for item in first.files
            if item.relative_path == "src/hello.py"
        )

        second_file = next(
            item
            for item in second.files
            if item.relative_path == "src/hello.py"
        )

        assert first_file.content_hash == (
            second_file.content_hash
        )

    def test_file_id_is_deterministic(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        first = ingestor.ingest(tmp_path)
        second = ingestor.ingest(tmp_path)

        first_file = next(
            item
            for item in first.files
            if item.relative_path == "src/hello.py"
        )

        second_file = next(
            item
            for item in second.files
            if item.relative_path == "src/hello.py"
        )

        assert first_file.file_id == second_file.file_id

    def test_documentation_is_identified(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        readme = next(
            item
            for item in snapshot.files
            if item.relative_path == "README.md"
        )

        assert readme.metadata["is_documentation"] is True

    def test_failed_file_can_be_non_fatal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        original_read_text = Path.read_text

        def failing_read_text(
            self: Path,
            *args: object,
            **kwargs: object,
        ) -> str:
            if self.name == "hello.py":
                raise OSError("simulated read failure")

            return original_read_text(
                self,
                *args,
                **kwargs,
            )

        monkeypatch.setattr(
            Path,
            "read_text",
            failing_read_text,
        )

        snapshot = ingestor.ingest(tmp_path)

        assert snapshot.failed_files == 1
        assert all(
            item.relative_path != "src/hello.py"
            for item in snapshot.files
        )

    def test_failed_file_can_be_fatal(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        create_repository(tmp_path)

        config = RepositoryIngestionConfig(
            fail_fast=True,
        )

        ingestor = RepositoryIngestor(
            config=config,
        )

        original_read_text = Path.read_text

        def failing_read_text(
            self: Path,
            *args: object,
            **kwargs: object,
        ) -> str:
            if self.name == "hello.py":
                raise OSError("simulated read failure")

            return original_read_text(
                self,
                *args,
                **kwargs,
            )

        monkeypatch.setattr(
            Path,
            "read_text",
            failing_read_text,
        )

        with pytest.raises(SourceFileReadError):
            ingestor.ingest(tmp_path)

    def test_snapshot_metadata_is_populated(
        self,
        tmp_path: Path,
    ) -> None:
        create_repository(tmp_path)

        ingestor = RepositoryIngestor()

        snapshot = ingestor.ingest(tmp_path)

        assert (
            snapshot.metadata["filter_accepted_files"]
            == 3
        )

        assert (
            snapshot.metadata["filter_skipped_files"]
            == 1
        )

        assert (
            snapshot.metadata["ingestion_failed_files"]
            == 0
        )