"""Contract tests for analysis history, detail, and debug read APIs."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.routes as routes_module
from app.main import create_app
from app.store import AnalysisSummary, StorageSettings, generate_analysis_id
from boxcheck.types import GroundTruthLabel


def _client(tmp_path: Path, *, raise_server_exceptions: bool = True) -> TestClient:
    return TestClient(
        create_app(StorageSettings(tmp_path / "data")),
        raise_server_exceptions=raise_server_exceptions,
    )


def _summary(tmp_path: Path, analysis_id: str, *, verdict: str = "3d") -> AnalysisSummary:
    return AnalysisSummary(
        id=analysis_id,
        created_at=datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc),
        label=GroundTruthLabel.REAL,
        phone_model="Pixel 7a",
        attempt=2 if verdict == "review" else 1,
        verdict=verdict,
        score=0.81,
        confidence=0.74,
        video_path=tmp_path / "secret" / f"{analysis_id}.mp4",
        result_path=tmp_path / "secret" / f"{analysis_id}.json",
        debug_directory=tmp_path / "secret" / analysis_id,
    )


def _result(analysis_id: str, *, verdict: str = "3d") -> dict:
    return {
        "id": analysis_id,
        "created_at": "2026-10-30T10:00:00+00:00",
        "attempt": 2 if verdict == "review" else 1,
        "verdict": verdict,
        "score": 0.81,
        "confidence": 0.74,
        "reason": "Two surfaces detected.",
        "metrics": {
            "frames_sampled": 12,
            "frames_used": 10,
            "pairs_used": 0,
            "median_matches": 0.0,
            "median_movement_px": 0.0,
            "median_h_inlier_ratio": 0.0,
            "median_f_inlier_ratio": 0.0,
            "second_plane_pair_fraction": 0.0,
            "median_parallax_px": 0.0,
            "median_brightness": 120.0,
            "median_blur": 90.0,
        },
        "pairs": [],
        "processing_ms": 120,
        "label": "real",
        "phone_model": "Pixel 7a",
        "debug_frames": ["debug/secret-frame.jpg"],
        "video_path": "uploads/secret-video.mp4",
        "result_path": "results/secret-result.json",
        "debug_directory": "debug/secret",
    }


def test_history_returns_public_summaries_and_pagination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first_id = generate_analysis_id()
    second_id = generate_analysis_id()
    captured: dict[str, object] = {}

    def fake_list_analyses(limit, offset, settings):
        captured.update(limit=limit, offset=offset, settings=settings)
        return [
            _summary(tmp_path, first_id, verdict="review"),
            _summary(tmp_path, second_id),
        ]

    monkeypatch.setattr(routes_module, "list_analyses", fake_list_analyses)

    with _client(tmp_path) as client:
        response = client.get("/api/v1/analyses", params={"limit": 20, "offset": 40})

    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 20
    assert body["offset"] == 40
    assert [item["id"] for item in body["items"]] == [first_id, second_id]
    assert body["items"][0]["verdict"] == "review"
    assert body["items"][0]["detail_url"] == f"/api/v1/analyses/{first_id}"
    assert set(body["items"][0]).isdisjoint(
        {"video_path", "result_path", "debug_directory"}
    )
    assert captured["limit"] == 20
    assert captured["offset"] == 40
    assert captured["settings"].data_directory == tmp_path / "data"  # type: ignore[union-attr]


def test_history_defaults_and_empty_page(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_list_analyses(limit, offset, settings):
        captured.update(limit=limit, offset=offset)
        return []

    monkeypatch.setattr(routes_module, "list_analyses", fake_list_analyses)

    with _client(tmp_path) as client:
        response = client.get("/api/v1/analyses")

    assert response.status_code == 200
    assert response.json() == {"items": [], "limit": 100, "offset": 0}
    assert captured == {"limit": 100, "offset": 0}


@pytest.mark.parametrize(
    "query",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"limit": "many"},
    ],
)
def test_history_rejects_invalid_pagination(tmp_path: Path, query: dict) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/v1/analyses", params=query)

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed."}


def test_detail_returns_complete_public_result(tmp_path: Path, monkeypatch) -> None:
    analysis_id = generate_analysis_id()
    captured: dict[str, object] = {}

    def fake_get_result(requested_id, settings):
        captured.update(analysis_id=requested_id, settings=settings)
        return _result(requested_id, verdict="review")

    monkeypatch.setattr(routes_module, "get_result", fake_get_result)

    with _client(tmp_path) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == analysis_id
    assert body["verdict"] == "review"
    assert body["metrics"]["frames_used"] == 10
    assert body["debug_frames"] == [
        f"/api/v1/analyses/{analysis_id}/debug/0.jpg"
    ]
    assert set(body).isdisjoint({"video_path", "result_path", "debug_directory"})
    assert captured["analysis_id"] == analysis_id


def test_unknown_detail_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(routes_module, "get_result", lambda *args, **kwargs: None)
    analysis_id = generate_analysis_id()

    with _client(tmp_path) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Analysis not found."}


@pytest.mark.parametrize("analysis_id", ["not-a-uuid", "../secret", "00000000-0000-0000-0000-000000000000"])
def test_detail_rejects_invalid_or_non_v4_id(tmp_path: Path, analysis_id: str) -> None:
    with _client(tmp_path) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}")

    assert response.status_code in {404, 422}
    if analysis_id != "../secret":
        assert response.status_code == 422
        assert response.json() == {"detail": "Request validation failed."}


def test_debug_endpoint_returns_verified_jpeg(tmp_path: Path, monkeypatch) -> None:
    analysis_id = generate_analysis_id()
    frame = tmp_path / "verified" / "pair.jpg"
    frame.parent.mkdir()
    jpeg_bytes = b"\xff\xd8test-jpeg\xff\xd9"
    frame.write_bytes(jpeg_bytes)
    captured: dict[str, object] = {}

    def fake_get_debug_frame(requested_id, frame_number, settings):
        captured.update(
            analysis_id=requested_id,
            frame_number=frame_number,
            settings=settings,
        )
        return frame

    monkeypatch.setattr(routes_module, "get_debug_frame", fake_get_debug_frame)

    with _client(tmp_path) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}/debug/3.jpg")

    assert response.status_code == 200
    assert response.content == jpeg_bytes
    assert response.headers["content-type"].startswith("image/jpeg")
    assert captured["analysis_id"] == analysis_id
    assert captured["frame_number"] == 3


def test_missing_debug_frame_returns_404(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(routes_module, "get_debug_frame", lambda *args, **kwargs: None)
    analysis_id = generate_analysis_id()

    with _client(tmp_path) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}/debug/9.jpg")

    assert response.status_code == 404
    assert response.json() == {"detail": "Analysis or debug frame not found."}


@pytest.mark.parametrize("frame", ["-1", "abc", "1.5"])
def test_debug_frame_rejects_invalid_index(tmp_path: Path, frame: str) -> None:
    analysis_id = generate_analysis_id()

    with _client(tmp_path) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}/debug/{frame}.jpg")

    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed."}


def test_storage_safety_failure_is_sanitized_500(tmp_path: Path, monkeypatch) -> None:
    analysis_id = generate_analysis_id()

    def unsafe_frame(*args, **kwargs):
        raise ValueError("Stored debug frame escapes C:/private/data")

    monkeypatch.setattr(routes_module, "get_debug_frame", unsafe_frame)

    with _client(tmp_path, raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/analyses/{analysis_id}/debug/0.jpg")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "private" not in response.text


def test_openapi_documents_all_read_routes(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/analyses" in paths
    assert "/api/v1/analyses/{analysis_id}" in paths
    assert "/api/v1/analyses/{analysis_id}/debug/{frame_number}.jpg" in paths
    assert "404" in paths["/api/v1/analyses/{analysis_id}"]["get"]["responses"]
