"""Contract and integration tests for the analysis rerun endpoint."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

import app.routes as routes_module
import app.store as store_module
from app.main import create_app
from app.policies import apply_review_policy
from app.store import StoredAnalysis, StorageSettings, generate_analysis_id
from boxcheck.types import AnalysisMetrics, AnalysisResult, GroundTruthLabel, Verdict


def _client(
    settings: StorageSettings,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    return TestClient(
        create_app(settings),
        raise_server_exceptions=raise_server_exceptions,
    )


def _record(settings: StorageSettings, *, verdict: str = "3d") -> StoredAnalysis:
    analysis_id = generate_analysis_id()
    result = AnalysisResult(
        verdict=(Verdict.INSUFFICIENT if verdict == "review" else Verdict(verdict)),
        score=0.5,
        confidence=0.6,
        reason="Rerun completed.",
        metrics=AnalysisMetrics(frames_sampled=8, frames_used=6),
        processing_ms=100,
        label=GroundTruthLabel.REAL,
        phone_model="Pixel 7a",
    )
    return StoredAnalysis(
        id=analysis_id,
        created_at=datetime.now(timezone.utc),
        attempt=2,
        video_path=settings.uploads_directory / f"{analysis_id}.mp4",
        result_path=settings.results_directory / f"{analysis_id}.json",
        debug_directory=settings.debug_directory / analysis_id,
        result=result,
        verdict=verdict,
    )


def test_rerun_endpoint_returns_new_public_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = StorageSettings(tmp_path / "data")
    source_id = generate_analysis_id()
    rerun = _record(settings, verdict="review")
    captured: dict[str, object] = {}

    def fake_rerun_analysis(analysis_id, settings, result_policy):
        captured.update(
            analysis_id=analysis_id,
            settings=settings,
            result_policy=result_policy,
        )
        return rerun

    monkeypatch.setattr(routes_module, "rerun_analysis", fake_rerun_analysis)

    with _client(settings) as client:
        response = client.post(f"/api/v1/analyses/{source_id}/rerun")

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == rerun.id
    assert body["id"] != source_id
    assert body["verdict"] == "review"
    assert body["label"] == "real"
    assert body["phone_model"] == "Pixel 7a"
    assert body["attempt"] == 2
    assert captured["analysis_id"] == source_id
    assert captured["settings"] is settings
    assert captured["result_policy"] is apply_review_policy
    assert set(body).isdisjoint({"video_path", "result_path", "debug_directory"})


def test_unknown_rerun_source_returns_404(tmp_path: Path, monkeypatch) -> None:
    settings = StorageSettings(tmp_path / "data")
    monkeypatch.setattr(routes_module, "rerun_analysis", lambda *args, **kwargs: None)

    with _client(settings) as client:
        response = client.post(
            f"/api/v1/analyses/{generate_analysis_id()}/rerun"
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Analysis not found."}


def test_rerun_rejects_invalid_or_non_v4_id(tmp_path: Path) -> None:
    settings = StorageSettings(tmp_path / "data")

    with _client(settings) as client:
        invalid = client.post("/api/v1/analyses/not-a-uuid/rerun")
        non_v4 = client.post(
            "/api/v1/analyses/00000000-0000-0000-0000-000000000000/rerun"
        )

    for response in (invalid, non_v4):
        assert response.status_code == 422
        assert response.json() == {"detail": "Request validation failed."}


def test_rerun_uses_threadpool(tmp_path: Path, monkeypatch) -> None:
    settings = StorageSettings(tmp_path / "data")
    source_id = generate_analysis_id()
    rerun = _record(settings)
    calls: list[object] = []
    monkeypatch.setattr(routes_module, "rerun_analysis", lambda *args, **kwargs: rerun)

    async def recording_threadpool(operation):
        calls.append(operation)
        return operation()

    monkeypatch.setattr(routes_module, "run_in_threadpool", recording_threadpool)

    with _client(settings) as client:
        response = client.post(f"/api/v1/analyses/{source_id}/rerun")

    assert response.status_code == 201
    assert len(calls) == 1


def test_rerun_failure_is_sanitized_500(tmp_path: Path, monkeypatch) -> None:
    settings = StorageSettings(tmp_path / "data")

    def fail_rerun(*args, **kwargs):
        raise RuntimeError("failed at C:/private/results")

    monkeypatch.setattr(routes_module, "rerun_analysis", fail_rerun)

    with _client(settings, raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/v1/analyses/{generate_analysis_id()}/rerun"
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "private" not in response.text


def test_storage_backed_rerun_preserves_original_and_applies_review_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = StorageSettings(tmp_path / "data")
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"original video bytes")

    def fake_analyze_video(
        video_path,
        label=None,
        phone_model=None,
        debug_output_directory=None,
    ) -> AnalysisResult:
        debug_frame = Path(debug_output_directory) / "pair_000.jpg"
        debug_frame.write_bytes(b"jpeg")
        return AnalysisResult(
            verdict=Verdict.INSUFFICIENT,
            score=0.5,
            confidence=0.3,
            reason="Not enough movement.",
            metrics=AnalysisMetrics(frames_sampled=8, frames_used=6),
            processing_ms=100,
            label=label,
            phone_model=phone_model,
            debug_frames=[debug_frame],
        )

    monkeypatch.setattr(store_module, "analyze_video", fake_analyze_video)
    original = store_module.create_analysis(
        source,
        label="real",
        phone_model="Pixel 7a",
        attempt=2,
        settings=settings,
    )
    original_video = original.video_path.read_bytes()
    original_json = original.result_path.read_bytes()

    with _client(settings) as client:
        response = client.post(f"/api/v1/analyses/{original.id}/rerun")

    assert response.status_code == 201
    body = response.json()
    assert body["id"] != original.id
    assert body["verdict"] == "review"
    assert body["label"] == "real"
    assert body["phone_model"] == "Pixel 7a"
    assert body["attempt"] == 2
    assert original.video_path.read_bytes() == original_video
    assert original.result_path.read_bytes() == original_json
    rerun = store_module.get_analysis(body["id"], settings)
    assert rerun is not None
    assert rerun.video_path.read_bytes() == original_video
    assert rerun.verdict == "review"
    assert len(store_module.list_analyses(settings=settings)) == 2


def test_openapi_documents_rerun_as_created_resource(tmp_path: Path) -> None:
    settings = StorageSettings(tmp_path / "data")

    with _client(settings) as client:
        operation = client.get("/openapi.json").json()["paths"][
            "/api/v1/analyses/{analysis_id}/rerun"
        ]["post"]

    assert "201" in operation["responses"]
    assert "404" in operation["responses"]
    assert "422" in operation["responses"]
