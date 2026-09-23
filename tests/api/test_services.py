"""Tests for SecureCodeRAG API services."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.services import (
    RepositoryIngestionService,
    RepositoryIngestionServiceError,
)
from src.ingestion.repository_ingestor import RepositoryIngestor


class TestRepositoryIngestionService:
    """Tests for repository ingestion service."""

    def test_ingests_repository(
        self,
        tmp_path: Path,
    ) -> None:
        source_file = tmp_path / "hello.py"
        source_file.write_text(
            "def hello():\n    return 'hello'\n",
            encoding="utf-8",
        )

        service = RepositoryIngestionService()

        result = service.ingest(str(tmp_path))

        assert result.repository_id.startswith("repo-")
        assert result.root_path == tmp_path.resolve().as_posix()
        assert result.file_count >= 1
        assert result.ingested_file_count == 1
        assert result.skipped_file_count >= 0
        assert result.failed_file_count == 0

    def test_empty_path_is_rejected(self) -> None:
        service = RepositoryIngestionService()

        with pytest.raises(RepositoryIngestionServiceError):
            service.ingest("")

    def test_whitespace_path_is_rejected(self) -> None:
        service = RepositoryIngestionService()

        with pytest.raises(RepositoryIngestionServiceError):
            service.ingest("   ")

    def test_non_string_path_is_rejected(self) -> None:
        service = RepositoryIngestionService()

        with pytest.raises(RepositoryIngestionServiceError):
            service.ingest(None)  # type: ignore[arg-type]

    def test_ingestor_failure_is_translated(
        self,
        tmp_path: Path,
    ) -> None:
        class FailingIngestor(RepositoryIngestor):
            def ingest(
                self,
                repository_path: str | Path,
            ):
                raise RuntimeError("test failure")

        service = RepositoryIngestionService(
            ingestor=FailingIngestor(),
        )

        with pytest.raises(RepositoryIngestionServiceError):
            service.ingest(str(tmp_path))