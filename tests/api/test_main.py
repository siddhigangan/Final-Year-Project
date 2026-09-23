"""Tests for the SecureCodeRAG FastAPI application."""

from __future__ import annotations

from pathlib import Path

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


class TestRepositoryIngestionEndpoint:
    """Tests for repository ingestion."""

    def test_ingest_repository(self, tmp_path: Path) -> None:
        source_file = tmp_path / "hello.py"

        source_file.write_text(
            "def hello():\n"
            "    return 'hello'\n",
            encoding="utf-8",
        )

        response = client.post(
            "/repositories/ingest",
            json={
                "repository_path": str(tmp_path),
                "include_content": True,
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["repository_id"].startswith("repo-")
        assert payload["root_path"] == tmp_path.resolve().as_posix()
        assert payload["file_count"] >= 1
        assert payload["ingested_file_count"] == 1
        assert payload["failed_file_count"] == 0

    def test_ingest_empty_path_is_rejected(self) -> None:
        response = client.post(
            "/repositories/ingest",
            json={
                "repository_path": "",
            },
        )

        assert response.status_code == 422

    def test_ingest_missing_path_is_rejected(self) -> None:
        response = client.post(
            "/repositories/ingest",
            json={},
        )

        assert response.status_code == 422

    def test_ingest_nonexistent_repository_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        missing_path = tmp_path / "does-not-exist"

        response = client.post(
            "/repositories/ingest",
            json={
                "repository_path": str(missing_path),
            },
        )

        assert response.status_code == 400

        payload = response.json()

        assert payload["error"] == "RepositoryIngestionError"