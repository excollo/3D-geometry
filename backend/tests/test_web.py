"""Tests for the no-build laptop upload page."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.store import StorageSettings


def _page(tmp_path: Path):
    application = create_app(StorageSettings(tmp_path / "data"))
    with TestClient(application) as client:
        return client.get("/")


def test_root_serves_laptop_page_as_html(tmp_path: Path) -> None:
    response = _page(tmp_path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>3D Box Checker</title>" in response.text
    assert "Is it really a box?" in response.text


def test_page_contains_complete_upload_form(tmp_path: Path) -> None:
    html = _page(tmp_path).text

    assert 'id="analysis-form"' in html
    assert 'name="video"' in html
    assert 'accept=".mp4,.mov,video/mp4,video/quicktime"' in html
    assert 'name="label"' in html
    assert 'value="flat_printout"' in html
    assert 'value="folded_printout"' in html
    assert 'value="bad_recording"' in html
    assert 'name="phone_model"' in html
    assert 'name="attempt"' in html
    assert 'min="1"' in html
    assert 'value="1"' in html


def test_page_calls_the_public_upload_api_with_form_data(tmp_path: Path) -> None:
    html = _page(tmp_path).text

    assert 'new FormData()' in html
    assert 'formData.append("video", video)' in html
    assert 'formData.append("attempt", attempt)' in html
    assert 'fetch("/api/v1/analyze"' in html
    assert 'method: "POST"' in html
    assert "Content-Type" not in html


def test_page_has_loading_error_and_accessible_status_behavior(tmp_path: Path) -> None:
    html = _page(tmp_path).text

    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert 'submitButton.disabled = isBusy' in html
    assert 'form.setAttribute("aria-busy", String(isBusy))' in html
    assert 'if (!response.ok) throw new Error' in html
    assert 'status.classList.add("error")' in html
    assert 'finally {' in html


def test_page_renders_all_verdicts_metrics_and_debug_images(tmp_path: Path) -> None:
    html = _page(tmp_path).text

    for verdict in ("3d", "flat", "insufficient", "review"):
        assert f'data-verdict="{verdict}"' in html
    assert 'Object.entries(data.metrics)' in html
    assert 'data.debug_frames.forEach' in html
    assert 'image.src = url' in html
    assert 'image.alt = `Annotated analysis frame ${index + 1}`' in html
    assert 'debugSection.hidden = data.debug_frames.length === 0' in html


def test_page_uses_safe_dom_text_instead_of_html_injection(tmp_path: Path) -> None:
    html = _page(tmp_path).text

    assert ".textContent" in html
    assert "innerHTML" not in html


def test_root_page_is_not_added_to_openapi(tmp_path: Path) -> None:
    application = create_app(StorageSettings(tmp_path / "data"))

    with TestClient(application) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert "/api/v1/analyze" in paths


def test_html_is_declared_as_installable_package_data() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in text
    assert 'app = ["web/*.html"]' in text
