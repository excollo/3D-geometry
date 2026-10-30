"""Tests for FastAPI application construction and shared server behavior."""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.errors import (
    ApiError,
    BadRequestError,
    NotFoundError,
    UnprocessableEntityError,
)
from app.store import StorageSettings


def _test_app(tmp_path: Path):
    settings = StorageSettings(tmp_path / "isolated-data")
    return main_module.create_app(settings), settings


def test_create_app_configures_metadata_and_injected_storage(tmp_path: Path) -> None:
    application, settings = _test_app(tmp_path)

    assert application.title == "3D Box Checker API"
    assert application.version == "0.1.0"
    assert "Local-network API" in application.description
    assert application.state.storage_settings is settings
    assert not settings.data_directory.exists()


def test_lifespan_initializes_directories_and_database(tmp_path: Path) -> None:
    application, settings = _test_app(tmp_path)

    with TestClient(application):
        assert settings.uploads_directory.is_dir()
        assert settings.results_directory.is_dir()
        assert settings.debug_directory.is_dir()
        assert settings.database_path.is_file()


def test_lifespan_initialization_is_safe_to_repeat(tmp_path: Path) -> None:
    application, settings = _test_app(tmp_path)

    with TestClient(application):
        marker = settings.uploads_directory / "existing.mp4"
        marker.write_bytes(b"existing")
    with TestClient(application):
        assert marker.read_bytes() == b"existing"


def test_openapi_and_interactive_docs_are_available(tmp_path: Path) -> None:
    application, _ = _test_app(tmp_path)

    with TestClient(application) as client:
        openapi = client.get("/openapi.json")
        docs = client.get("/docs")

    assert openapi.status_code == 200
    assert openapi.json()["info"]["title"] == "3D Box Checker API"
    assert openapi.json()["info"]["version"] == "0.1.0"
    assert docs.status_code == 200
    assert "text/html" in docs.headers["content-type"]


@pytest.mark.parametrize(
    "error_type,status_code,detail",
    [
        (BadRequestError, 400, "Bad video."),
        (NotFoundError, 404, "Missing analysis."),
        (UnprocessableEntityError, 422, "Invalid ID."),
    ],
)
def test_api_errors_return_stable_json(
    tmp_path: Path,
    error_type: type[ApiError],
    status_code: int,
    detail: str,
) -> None:
    application, _ = _test_app(tmp_path)

    @application.get("/test-error")
    async def test_error_route():
        raise error_type(detail)

    with TestClient(application) as client:
        response = client.get("/test-error")

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}
    assert response.headers["content-type"].startswith("application/json")


def test_request_validation_returns_stable_422_json(tmp_path: Path) -> None:
    application, _ = _test_app(tmp_path)

    @application.get("/validated")
    async def validated_route(value: int):
        return {"value": value}

    with TestClient(application) as client:
        response = client.get("/validated", params={"value": "not-an-integer"})

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed."}


def test_unknown_route_uses_stable_error_shape(tmp_path: Path) -> None:
    application, _ = _test_app(tmp_path)

    with TestClient(application) as client:
        response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_unexpected_error_is_logged_and_sanitized(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    application, _ = _test_app(tmp_path)

    @application.get("/crash")
    async def crash_route():
        raise RuntimeError("sensitive C:/private/result.json")

    caplog.set_level(logging.ERROR, logger="app.main")
    with TestClient(application, raise_server_exceptions=False) as client:
        response = client.get("/crash")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "private" not in response.text
    assert "Unhandled API error for GET /crash" in caplog.text


def test_create_app_rejects_invalid_storage_settings() -> None:
    with pytest.raises(TypeError, match="StorageSettings"):
        main_module.create_app(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("detail", ["", "   "])
def test_api_error_rejects_blank_detail(detail: str) -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        BadRequestError(detail)


def test_module_exports_ready_to_run_application() -> None:
    assert main_module.app.title == "3D Box Checker API"
    assert isinstance(main_module.app.state.storage_settings, StorageSettings)
