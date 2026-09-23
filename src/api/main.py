"""FastAPI application for SecureCodeRAG."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.schemas import (
    APIErrorResponse,
    HealthResponse,
    RepositoryIngestRequest,
    RepositoryIngestResponse,
)
from src.api.services import (
    RepositoryIngestionService,
    RepositoryIngestionServiceError,
)

APP_TITLE = "SecureCodeRAG API"
APP_VERSION = "0.1.0"


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "Security-aware code retrieval and generation API "
        "for SecureCodeRAG."
    ),
)

repository_ingestion_service = RepositoryIngestionService()


@app.get(
    "/health",
    response_model=None,
    tags=["system"],
)
def health() -> dict[str, str]:
    """Return the API health status."""
    response = HealthResponse(
        status="ok",
        service=APP_TITLE,
        version=APP_VERSION,
    )

    return response.to_dict()


@app.get(
    "/",
    response_model=None,
    tags=["system"],
)
def root() -> dict[str, str]:
    """Return basic API information."""
    return {
        "service": APP_TITLE,
        "version": APP_VERSION,
        "status": "ok",
        "docs": "/docs",
    }


@app.post(
    "/repositories/ingest",
    response_model=RepositoryIngestResponse,
    tags=["repositories"],
)
def ingest_repository(
    request: RepositoryIngestRequest,
) -> RepositoryIngestResponse:
    """Ingest an authorized local repository."""
    summary = repository_ingestion_service.ingest(
        repository_path=request.repository_path,
    )

    return RepositoryIngestResponse(
        repository_id=summary.repository_id,
        root_path=summary.root_path,
        file_count=summary.file_count,
        ingested_file_count=summary.ingested_file_count,
        skipped_file_count=summary.skipped_file_count,
        failed_file_count=summary.failed_file_count,
    )


@app.exception_handler(RepositoryIngestionServiceError)
async def repository_ingestion_error_handler(
    _request: object,
    exc: RepositoryIngestionServiceError,
) -> JSONResponse:
    """Convert ingestion service failures into API errors."""
    response = APIErrorResponse(
        error="RepositoryIngestionError",
        message=str(exc),
    )

    return JSONResponse(
        status_code=400,
        content=response.to_dict(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(
    _request: object,
    exc: ValueError,
) -> JSONResponse:
    """Convert validation-related ValueErrors into API responses."""
    response = APIErrorResponse(
        error="ValidationError",
        message=str(exc),
    )

    return JSONResponse(
        status_code=400,
        content=response.to_dict(),
    )