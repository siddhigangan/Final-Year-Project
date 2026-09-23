"""Tests for SecureCodeRAG API schemas."""

from __future__ import annotations

import pytest

from src.api.schemas import (
    APIErrorResponse,
    APISchemaError,
    HealthResponse,
    RepositoryIngestRequest,
    RepositoryIngestResponse,
    SearchRequest,
    SearchResultItem,
    SecurityAnalysisRequest,
    SecurityAnalysisResponse,
)


class TestHealthResponse:
    """Tests for the health response."""

    def test_default_values(self) -> None:
        response = HealthResponse()

        assert response.status == "ok"
        assert response.service == "SecureCodeRAG"
        assert response.version == "0.1.0"

    def test_to_dict(self) -> None:
        response = HealthResponse()

        assert response.to_dict() == {
            "status": "ok",
            "service": "SecureCodeRAG",
            "version": "0.1.0",
        }


class TestRepositoryIngestRequest:
    """Tests for repository ingestion requests."""

    def test_valid_request(self) -> None:
        request = RepositoryIngestRequest(
            repository_path="C:/repositories/example",
        )

        assert request.repository_path == (
            "C:/repositories/example"
        )
        assert request.include_content is True

    def test_empty_path_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            RepositoryIngestRequest(repository_path="")

    def test_whitespace_path_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            RepositoryIngestRequest(repository_path="   ")


class TestRepositoryIngestResponse:
    """Tests for repository ingestion responses."""

    def test_valid_response(self) -> None:
        response = RepositoryIngestResponse(
            repository_id="repo-123",
            root_path="C:/repositories/example",
            file_count=10,
            ingested_file_count=8,
            skipped_file_count=1,
            failed_file_count=1,
        )

        payload = response.to_dict()

        assert payload["repository_id"] == "repo-123"
        assert payload["file_count"] == 10
        assert payload["ingested_file_count"] == 8

    def test_negative_file_count_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            RepositoryIngestResponse(
                repository_id="repo-123",
                root_path="C:/repositories/example",
                file_count=-1,
                ingested_file_count=0,
                skipped_file_count=0,
                failed_file_count=0,
            )


class TestSearchRequest:
    """Tests for search requests."""

    def test_defaults(self) -> None:
        request = SearchRequest(
            query="find authentication code",
        )

        assert request.top_k == 5
        assert request.repository_id is None

    def test_custom_top_k(self) -> None:
        request = SearchRequest(
            query="find authentication code",
            top_k=10,
            repository_id="repo-123",
        )

        assert request.top_k == 10
        assert request.repository_id == "repo-123"

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            SearchRequest(query="")

    def test_zero_top_k_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            SearchRequest(
                query="authentication",
                top_k=0,
            )

    def test_excessive_top_k_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            SearchRequest(
                query="authentication",
                top_k=101,
            )


class TestSearchResultItem:
    """Tests for retrieved result serialization."""

    def test_to_dict(self) -> None:
        result = SearchResultItem(
            chunk_id="chunk-1",
            source_file_id="file-1",
            repository_id="repo-1",
            content="def hello():\n    return 'hello'",
            language="python",
            start_line=1,
            end_line=2,
            retrieval_score=0.95,
            rank=1,
            trust="trusted",
        )

        payload = result.to_dict()

        assert payload["chunk_id"] == "chunk-1"
        assert payload["language"] == "python"
        assert payload["retrieval_score"] == 0.95


class TestSecurityAnalysisRequest:
    """Tests for security analysis requests."""

    def test_valid_request(self) -> None:
        request = SecurityAnalysisRequest(
            query="How should this function be used?",
            context="def hello():\n    return 'hello'",
        )

        assert request.query
        assert request.context

    def test_empty_query_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            SecurityAnalysisRequest(
                query="",
                context="some code",
            )

    def test_empty_context_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            SecurityAnalysisRequest(
                query="analyze this",
                context="",
            )


class TestSecurityAnalysisResponse:
    """Tests for security analysis responses."""

    def test_valid_response(self) -> None:
        response = SecurityAnalysisResponse(
            decision="PASS",
            risk_level="LOW",
            finding_count=0,
            context_safe_to_use=True,
        )

        payload = response.to_dict()

        assert payload["decision"] == "PASS"
        assert payload["risk_level"] == "LOW"
        assert payload["finding_count"] == 0
        assert payload["context_safe_to_use"] is True

    def test_negative_finding_count_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            SecurityAnalysisResponse(
                decision="PASS",
                risk_level="LOW",
                finding_count=-1,
                context_safe_to_use=True,
            )


class TestAPIErrorResponse:
    """Tests for standardized API errors."""

    def test_to_dict(self) -> None:
        response = APIErrorResponse(
            error="ValidationError",
            message="Invalid request.",
            details={"field": "query"},
        )

        assert response.to_dict() == {
            "error": "ValidationError",
            "message": "Invalid request.",
            "details": {"field": "query"},
        }

    def test_empty_error_is_rejected(self) -> None:
        with pytest.raises(APISchemaError):
            APIErrorResponse(
                error="",
                message="Something went wrong.",
            )