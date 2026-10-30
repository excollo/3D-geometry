"""Contract tests for the Part 3 operator documentation."""

from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_documents_install_test_and_launch_commands() -> None:
    text = _readme()

    assert 'python -m pip install -e ".\\backend[dev]"' in text
    assert "python -m pytest" in text
    assert "python -m uvicorn app.main:app --app-dir backend" in text
    assert "--host 0.0.0.0 --port 8000" in text


def test_readme_documents_local_page_docs_and_openapi() -> None:
    text = _readme()

    assert "http://127.0.0.1:8000/" in text
    assert "http://127.0.0.1:8000/docs" in text
    assert "http://127.0.0.1:8000/openapi.json" in text


def test_readme_documents_every_public_route() -> None:
    text = _readme()

    for route in (
        "/api/v1/analyze",
        "/api/v1/analyses",
        "/api/v1/analyses/{id}",
        "/api/v1/analyses/{id}/debug/{n}.jpg",
        "/api/v1/analyses/{id}/rerun",
    ):
        assert route in text


def test_readme_contains_real_video_and_wifi_verification_steps() -> None:
    text = _readme()

    assert "curl.exe" in text
    assert 'name="video"' not in text  # Instructions use the public field, not HTML internals.
    assert '-F "video=@' in text
    assert "ipconfig" in text
    assert "http://LAPTOP_IP:8000/" in text
    assert "Windows Firewall" in text
    assert "Private networks" in text
    assert "client/AP isolation" in text


def test_readme_states_local_development_security_boundary() -> None:
    text = _readme()

    for limitation in (
        "no authentication",
        "authorization",
        "rate limiting",
        "HTTPS",
        "trusted private network",
    ):
        assert limitation in text
    assert "do not expose" in text


def test_readme_marks_part_three_implemented() -> None:
    text = _readme()

    assert "Parts 1, 2, and 3 are implemented" in text
    assert "Part 4 is the evaluation scorecard" in text
