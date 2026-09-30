"""Application services for SecureCodeRAG API."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.repository_ingestor import (
    RepositoryIngestionError,
    RepositoryIngestor,
)


class APIServiceError(RuntimeError):
    """Base exception for API service failures."""


class RepositoryIngestionServiceError(APIServiceError):
    """Raised when repository ingestion fails through the API."""


@dataclass(frozen=True)
class RepositoryIngestionSummary:
    """API-friendly summary of repository ingestion."""

    repository_id: str
    root_path: str
    file_count: int
    ingested_file_count: int
    skipped_file_count: int
    failed_file_count: int


class RepositoryIngestionService:
    """Application service for repository ingestion."""

    def __init__(
        self,
        ingestor: RepositoryIngestor | None = None,
    ) -> None:
        """Initialize the service."""
        self.ingestor = ingestor or RepositoryIngestor()

    def ingest(
        self,
        repository_path: str,
    ) -> RepositoryIngestionSummary:
        """
        Ingest a repository and return a compact API summary.

        The service delegates all repository processing to the existing
        RepositoryIngestor. It does not duplicate ingestion logic.
        """
        if not isinstance(repository_path, str):
            raise RepositoryIngestionServiceError(
                "repository_path must be a string"
            )

        if not repository_path.strip():
            raise RepositoryIngestionServiceError(
                "repository_path must not be empty"
            )

        try:
            snapshot = self.ingestor.ingest(repository_path)
        except RepositoryIngestionError as exc:
            raise RepositoryIngestionServiceError(
                f"Repository ingestion failed: {exc}"
            ) from exc
        except Exception as exc:
            raise RepositoryIngestionServiceError(
                "Unexpected repository ingestion failure"
            ) from exc

        return RepositoryIngestionSummary(
            repository_id=snapshot.repository_id,
            root_path=snapshot.root_path,
            file_count=snapshot.total_files,
            ingested_file_count=len(snapshot.files),
            skipped_file_count=snapshot.skipped_files,
            failed_file_count=snapshot.failed_files,
        )