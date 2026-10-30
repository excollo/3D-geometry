"""Cross-route integration and architecture contracts for the complete API."""

import ast
import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

import app.store as store_module
from app.main import create_app
from app.store import StorageSettings
from boxcheck.types import AnalysisMetrics, AnalysisResult, PairMetrics, Verdict


def _install_deterministic_checker(monkeypatch) -> None:
    def fake_analyze_video(
        video_path,
        label=None,
        phone_model=None,
        debug_output_directory=None,
    ) -> AnalysisResult:
        debug_frame = Path(debug_output_directory) / "pair_000.jpg"
        debug_frame.write_bytes(b"\xff\xd8annotated-frame\xff\xd9")
        return AnalysisResult(
            verdict=Verdict.INSUFFICIENT,
            score=0.48,
            confidence=0.31,
            reason="Not enough movement — move around the box.",
            metrics=AnalysisMetrics(
                frames_sampled=12,
                frames_used=10,
                pairs_used=1,
                median_matches=58,
                median_movement_px=3.2,
                median_h_inlier_ratio=0.82,
                median_f_inlier_ratio=0.71,
                second_plane_pair_fraction=0.0,
                median_parallax_px=1.1,
                median_brightness=121,
                median_blur=88,
            ),
            pairs=[
                PairMetrics(
                    first_frame_index=0,
                    second_frame_index=4,
                    time_gap_seconds=1.0,
                    match_count=58,
                    median_movement_px=3.2,
                    homography_inlier_ratio=0.82,
                    fundamental_inlier_ratio=0.71,
                    homography_gric=80.0,
                    fundamental_gric=93.0,
                    median_parallax_px=1.1,
                    pair_score=0.48,
                    low_motion=True,
                    low_parallax=True,
                    pure_rotation_risk=True,
                )
            ],
            processing_ms=125,
            label=label,
            phone_model=phone_model,
            debug_frames=[debug_frame],
        )

    monkeypatch.setattr(store_module, "analyze_video", fake_analyze_video)


def test_complete_api_workflow_is_consistent_across_http_files_and_sqlite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _install_deterministic_checker(monkeypatch)
    settings = StorageSettings(tmp_path / "data")
    application = create_app(settings)
    video_bytes = b"deterministic uploaded video"

    with TestClient(application) as client:
        page = client.get("/")
        upload = client.post(
            "/api/v1/analyze",
            files={"video": ("box.mp4", video_bytes, "video/mp4")},
            data={"label": "real", "phone_model": "Pixel 7a", "attempt": "2"},
        )

        assert page.status_code == 200
        assert 'fetch("/api/v1/analyze"' in page.text
        assert upload.status_code == 201
        uploaded = upload.json()
        original_id = uploaded["id"]
        assert uploaded["verdict"] == "review"
        assert uploaded["reason"] == "Not enough movement — move around the box."
        assert uploaded["metrics"]["frames_used"] == 10
        assert uploaded["pairs"][0]["pure_rotation_risk"] is True
        assert uploaded["debug_frames"] == [
            f"/api/v1/analyses/{original_id}/debug/0.jpg"
        ]

        original_video_path = settings.uploads_directory / f"{original_id}.mp4"
        original_result_path = settings.results_directory / f"{original_id}.json"
        original_video_bytes = original_video_path.read_bytes()
        original_result_bytes = original_result_path.read_bytes()
        assert original_video_bytes == video_bytes

        saved = json.loads(original_result_bytes)
        assert saved["id"] == original_id
        assert saved["verdict"] == uploaded["verdict"] == "review"
        assert saved["attempt"] == uploaded["attempt"] == 2
        assert saved["score"] == uploaded["score"]
        assert saved["confidence"] == uploaded["confidence"]

        with sqlite3.connect(settings.database_path) as connection:
            indexed = connection.execute(
                "SELECT verdict, attempt, label, phone_model, score, confidence "
                "FROM analyses WHERE id = ?",
                (original_id,),
            ).fetchone()
        assert indexed == ("review", 2, "real", "Pixel 7a", 0.48, 0.31)

        history = client.get("/api/v1/analyses")
        detail = client.get(f"/api/v1/analyses/{original_id}")
        debug = client.get(uploaded["debug_frames"][0])

        assert history.status_code == 200
        assert history.json()["items"][0]["id"] == original_id
        assert history.json()["items"][0]["verdict"] == "review"
        assert detail.status_code == 200
        assert detail.json() == uploaded
        assert debug.status_code == 200
        assert debug.headers["content-type"].startswith("image/jpeg")
        assert debug.content == b"\xff\xd8annotated-frame\xff\xd9"

        rerun = client.post(f"/api/v1/analyses/{original_id}/rerun")
        assert rerun.status_code == 201
        rerun_body = rerun.json()
        assert rerun_body["id"] != original_id
        assert rerun_body["verdict"] == "review"
        assert rerun_body["label"] == "real"
        assert rerun_body["phone_model"] == "Pixel 7a"
        assert rerun_body["attempt"] == 2

        history_after_rerun = client.get("/api/v1/analyses")
        assert history_after_rerun.status_code == 200
        assert {item["id"] for item in history_after_rerun.json()["items"]} == {
            original_id,
            rerun_body["id"],
        }

    assert original_video_path.read_bytes() == original_video_bytes
    assert original_result_path.read_bytes() == original_result_bytes
    assert (settings.uploads_directory / f"{rerun_body['id']}.mp4").read_bytes() == video_bytes
    with sqlite3.connect(settings.database_path) as connection:
        verdicts = connection.execute(
            "SELECT id, verdict FROM analyses ORDER BY created_at"
        ).fetchall()
    assert set(verdicts) == {
        (original_id, "review"),
        (rerun_body["id"], "review"),
    }

    public_text = json.dumps(
        {"upload": uploaded, "history": history_after_rerun.json(), "rerun": rerun_body}
    )
    assert str(tmp_path) not in public_text
    assert "video_path" not in public_text
    assert "result_path" not in public_text
    assert "debug_directory" not in public_text


def test_openapi_contains_every_versioned_api_contract(tmp_path: Path) -> None:
    application = create_app(StorageSettings(tmp_path / "data"))

    with TestClient(application) as client:
        paths = client.get("/openapi.json").json()["paths"]

    expected_methods = {
        "/api/v1/analyze": "post",
        "/api/v1/analyses": "get",
        "/api/v1/analyses/{analysis_id}": "get",
        "/api/v1/analyses/{analysis_id}/debug/{frame_number}.jpg": "get",
        "/api/v1/analyses/{analysis_id}/rerun": "post",
    }
    assert set(paths) == set(expected_methods)
    for path, method in expected_methods.items():
        operation = paths[path][method]
        assert "200" in operation["responses"] or "201" in operation["responses"]
        assert "422" in operation["responses"]
    assert "201" in paths["/api/v1/analyze"]["post"]["responses"]
    assert "201" in paths[
        "/api/v1/analyses/{analysis_id}/rerun"
    ]["post"]["responses"]


def test_checker_package_has_no_server_or_storage_imports() -> None:
    boxcheck_directory = Path(__file__).resolve().parents[1] / "boxcheck"
    forbidden_roots = {"app", "fastapi", "starlette"}
    violations: list[str] = []

    for source_path in boxcheck_directory.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            for module_name in imported:
                if module_name.split(".", 1)[0] in forbidden_roots:
                    violations.append(f"{source_path.name}: {module_name}")

    assert violations == []
