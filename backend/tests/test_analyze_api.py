"""Contract and lifecycle tests for POST /api/v1/analyze."""

from datetime import datetime, timezone
from pathlib import Path
from threading import get_ident

import pytest
from fastapi.testclient import TestClient

import app.routes as routes_module
from app.main import create_app
from app.policies import apply_review_policy
from app.store import StoredAnalysis, StorageSettings, generate_analysis_id
from boxcheck.types import AnalysisMetrics, AnalysisResult, Verdict
from boxcheck.video import VideoOpenError


def _install_fake_create_analysis(
    monkeypatch,
    captured: dict[str, object],
    *,
    checker_verdict: Verdict = Verdict.THREE_D,
) -> None:
    def fake_create_analysis(
        source_video_path,
        label=None,
        phone_model=None,
        attempt=1,
        settings=None,
        result_policy=None,
    ) -> StoredAnalysis:
        source = Path(source_video_path)
        captured.update(
            source=source,
            source_exists_during_analysis=source.is_file(),
            source_bytes=source.read_bytes(),
            label=label,
            phone_model=phone_model,
            attempt=attempt,
            settings=settings,
            result_policy=result_policy,
            worker_thread=get_ident(),
        )
        result = AnalysisResult(
            verdict=checker_verdict,
            score=0.81 if checker_verdict is Verdict.THREE_D else 0.5,
            confidence=0.74,
            reason="Test analysis completed.",
            metrics=AnalysisMetrics(frames_sampled=8, frames_used=6),
            processing_ms=120,
            label=label,
            phone_model=phone_model,
            debug_frames=[],
        )
        stored_verdict = (
            result_policy(result, attempt)
            if result_policy is not None
            else result.verdict.value
        )
        analysis_id = generate_analysis_id()
        return StoredAnalysis(
            id=analysis_id,
            created_at=datetime.now(timezone.utc),
            attempt=attempt,
            video_path=settings.uploads_directory / f"{analysis_id}.mp4",
            result_path=settings.results_directory / f"{analysis_id}.json",
            debug_directory=settings.debug_directory / analysis_id,
            result=result,
            verdict=stored_verdict,
        )

    monkeypatch.setattr(routes_module, "create_analysis", fake_create_analysis)


def _client(tmp_path: Path, *, raise_server_exceptions: bool = True) -> TestClient:
    application = create_app(StorageSettings(tmp_path / "data"))
    return TestClient(application, raise_server_exceptions=raise_server_exceptions)


def test_analyze_upload_returns_created_public_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_create_analysis(monkeypatch, captured)
    video_bytes = b"video-data" * 150_000

    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"video": ("box.MP4", video_bytes, "video/mp4")},
            data={"label": "real", "phone_model": "Pixel 7a", "attempt": "1"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["verdict"] == "3d"
    assert body["label"] == "real"
    assert body["phone_model"] == "Pixel 7a"
    assert body["attempt"] == 1
    assert captured["source_exists_during_analysis"] is True
    assert captured["source_bytes"] == video_bytes
    assert captured["result_policy"] is apply_review_policy
    assert captured["settings"].data_directory == tmp_path / "data"  # type: ignore[union-attr]
    assert not Path(captured["source"]).exists()  # type: ignore[arg-type]
    assert set(body).isdisjoint({"video_path", "result_path", "debug_directory"})


def test_second_insufficient_upload_returns_review(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_create_analysis(
        monkeypatch,
        captured,
        checker_verdict=Verdict.INSUFFICIENT,
    )

    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"video": ("box.mov", b"video", "video/quicktime")},
            data={"attempt": "2"},
        )

    assert response.status_code == 201
    assert response.json()["verdict"] == "review"
    assert response.json()["attempt"] == 2


def test_analysis_work_is_delegated_to_threadpool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    _install_fake_create_analysis(monkeypatch, captured)
    threadpool_calls: list[object] = []

    async def recording_threadpool(operation):
        threadpool_calls.append(operation)
        return operation()

    monkeypatch.setattr(routes_module, "run_in_threadpool", recording_threadpool)

    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"video": ("box.mp4", b"video", "video/mp4")},
        )

    assert response.status_code == 201
    assert len(threadpool_calls) == 1


@pytest.mark.parametrize(
    "files,data",
    [
        ({}, {}),
        ({"video": ("box.mp4", b"video", "video/mp4")}, {"attempt": "0"}),
        ({"video": ("box.mp4", b"video", "video/mp4")}, {"label": "unknown"}),
    ],
)
def test_analyze_upload_rejects_invalid_multipart_fields(
    tmp_path: Path,
    files: dict,
    data: dict,
) -> None:
    with _client(tmp_path) as client:
        response = client.post("/api/v1/analyze", files=files, data=data)

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed."}


def test_analyze_upload_rejects_unsupported_extension_before_storage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_create(*args, **kwargs):
        raise AssertionError("storage must not run")

    monkeypatch.setattr(routes_module, "create_analysis", unexpected_create)

    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"video": ("box.avi", b"video", "video/x-msvideo")},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unsupported video extension; use .mov, .mp4."
    }


def test_unreadable_video_returns_sanitized_400_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Path] = {}

    def fail_create(source_video_path, **kwargs):
        captured["source"] = Path(source_video_path)
        raise VideoOpenError(f"could not open {source_video_path}")

    monkeypatch.setattr(routes_module, "create_analysis", fail_create)

    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"video": ("broken.mp4", b"not-video", "video/mp4")},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The video could not be read. Upload a valid MP4 or MOV file."
    }
    assert not captured["source"].exists()
    assert str(captured["source"]) not in response.text


def test_unexpected_failure_returns_500_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Path] = {}

    def fail_create(source_video_path, **kwargs):
        captured["source"] = Path(source_video_path)
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(routes_module, "create_analysis", fail_create)

    with _client(tmp_path, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/analyze",
            files={"video": ("box.mp4", b"video", "video/mp4")},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert not captured["source"].exists()


def test_openapi_describes_multipart_upload_and_created_response(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        operation = client.get("/openapi.json").json()["paths"]["/api/v1/analyze"][
            "post"
        ]

    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert "201" in operation["responses"]
    assert "400" in operation["responses"]
