"""Tests for the SecureCodeRAG FastAPI application."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import APP_TITLE, APP_VERSION, app

client = TestClient(app)


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_api_information(self) -> None:
        response = client.get("/")

        assert response.status_code == 200

        payload = response.json()

        assert payload["service"] == APP_TITLE
        assert payload["version"] == APP_VERSION
        assert payload["status"] == "ok"
        assert payload["docs"] == "/docs"


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_returns_ok(self) -> None:
        response = client.get("/health")

        assert response.status_code == 200

        payload = response.json()

        assert payload["status"] == "ok"
        assert payload["service"] == APP_TITLE
        assert payload["version"] == APP_VERSION

    def test_health_response_is_json(self) -> None:
        response = client.get("/health")

        assert response.headers["content-type"].startswith(
            "application/json"
        )


class TestDocumentation:
    """Tests for automatically generated API documentation."""

    def test_openapi_is_available(self) -> None:
        response = client.get("/openapi.json")

        assert response.status_code == 200

        payload = response.json()

        assert payload["info"]["title"] == APP_TITLE
        assert payload["info"]["version"] == APP_VERSION

    def test_swagger_documentation_is_available(self) -> None:
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]