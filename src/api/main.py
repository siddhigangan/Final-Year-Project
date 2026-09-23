"""FastAPI application for SecureCodeRAG."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.schemas import APIErrorResponse, HealthResponse

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