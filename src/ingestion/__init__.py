"""
Ingestion module for SecureCodeRAG.

The ingestion layer is responsible for:

- repository discovery
- file filtering
- language detection
- source-file metadata
- deterministic repository snapshots

Parsing, chunking, retrieval, generation, and security analysis
are handled by later pipeline stages.
"""

from src.ingestion.models import (
    RepositorySnapshot,
    SourceFile,
)
from src.ingestion.repository_ingestor import (
    RepositoryIngestionConfig,
    RepositoryIngestionError,
    RepositoryIngestor,
    SourceFileReadError,
)

__all__ = [
    "RepositoryIngestionConfig",
    "RepositoryIngestionError",
    "RepositoryIngestor",
    "RepositorySnapshot",
    "SourceFile",
    "SourceFileReadError",
]