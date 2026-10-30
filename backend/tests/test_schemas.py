"""Tests for public API schemas and response conversion helpers."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.schemas import (
    AnalysisResponse,
    ErrorResponse,
    analysis_list_response,
    analysis_response_from_result,
    analysis_response_from_stored,
    analysis_summary_response,
)
from app.store import AnalysisSummary, StoredAnalysis, generate_analysis_id
from boxcheck.types import (
    AnalysisMetrics,
    AnalysisResult,
    GroundTruthLabel,
    PairMetrics,
    Verdict,
)


def _stored_analysis(tmp_path: Path, *, verdict: str = "3d") -> StoredAnalysis:
    analysis_id = generate_analysis_id()
    debug_directory = tmp_path / "debug" / analysis_id
    result = AnalysisResult(
        verdict=(Verdict.INSUFFICIENT if verdict == "review" else Verdict(verdict)),
        score=0.81,
        confidence=0.74,
        reason="Two surfaces at different angles.",
        metrics=AnalysisMetrics(
            frames_sampled=32,
            frames_used=27,
            pairs_used=1,
            median_matches=86,
            median_movement_px=41,
            median_h_inlier_ratio=0.62,
            median_f_inlier_ratio=0.79,
            second_plane_pair_fraction=0.64,
            median_parallax_px=6.1,
            median_brightness=124,
            median_blur=91,
        ),
        pairs=[
            PairMetrics(
                first_frame_index=0,
                second_frame_index=4,
                time_gap_seconds=1.0,
                match_count=86,
                median_movement_px=41,
                homography_inlier_ratio=0.62,
                fundamental_inlier_ratio=0.79,
                homography_gric=120.0,
                fundamental_gric=90.0,
                median_parallax_px=6.1,
                second_plane_found=True,
                second_plane_inliers=24,
                plane_normal_angle_degrees=38.0,
                box_corner_found=True,
                pair_score=0.81,
            )
        ],
        processing_ms=2800,
        label=GroundTruthLabel.REAL,
        phone_model="Pixel 7a",
        debug_frames=[debug_directory / "pair_000.jpg"],
    )
    return StoredAnalysis(
        id=analysis_id,
        created_at=datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc),
        attempt=2 if verdict == "review" else 1,
        video_path=tmp_path / "uploads" / f"{analysis_id}.mp4",
        result_path=tmp_path / "results" / f"{analysis_id}.json",
        debug_directory=debug_directory,
        result=result,
        verdict=verdict,
    )


def _summary(tmp_path: Path, *, verdict: str = "3d") -> AnalysisSummary:
    record = _stored_analysis(tmp_path, verdict=verdict)
    return AnalysisSummary(
        id=record.id,
        created_at=record.created_at,
        label=record.result.label,
        phone_model=record.result.phone_model,
        attempt=record.attempt,
        verdict=verdict,
        score=record.result.score,
        confidence=record.result.confidence,
        video_path=record.video_path,
        result_path=record.result_path,
        debug_directory=record.debug_directory,
    )


def test_full_stored_analysis_converts_to_public_response(tmp_path: Path) -> None:
    record = _stored_analysis(tmp_path)

    response = analysis_response_from_stored(record)

    assert response.id == record.id
    assert response.verdict == "3d"
    assert response.metrics.frames_used == 27
    assert response.pairs[0].match_count == 86
    assert response.debug_frames == [
        f"/api/v1/analyses/{record.id}/debug/0.jpg"
    ]


def test_review_is_public_while_checker_result_remains_insufficient(
    tmp_path: Path,
) -> None:
    record = _stored_analysis(tmp_path, verdict="review")

    response = analysis_response_from_stored(record)

    assert record.result.verdict is Verdict.INSUFFICIENT
    assert response.verdict == "review"
    assert response.attempt == 2


def test_full_response_never_serializes_internal_paths(tmp_path: Path) -> None:
    record = _stored_analysis(tmp_path)

    public_data = analysis_response_from_stored(record).model_dump(mode="json")
    serialized = str(public_data)

    assert "video_path" not in public_data
    assert "result_path" not in public_data
    assert "debug_directory" not in public_data
    assert str(tmp_path) not in serialized


def test_saved_result_mapping_uses_debug_count_not_saved_paths(tmp_path: Path) -> None:
    record = _stored_analysis(tmp_path)
    saved = record.to_dict()
    saved["debug_frames"] = ["secret/a.jpg", "secret/b.jpg"]

    response = analysis_response_from_result(saved)

    assert response.debug_frames == [
        f"/api/v1/analyses/{record.id}/debug/0.jpg",
        f"/api/v1/analyses/{record.id}/debug/1.jpg",
    ]


def test_summary_conversion_exposes_detail_url_without_paths(tmp_path: Path) -> None:
    summary = _summary(tmp_path, verdict="review")

    response = analysis_summary_response(summary)
    data = response.model_dump(mode="json")

    assert response.verdict == "review"
    assert response.detail_url == f"/api/v1/analyses/{summary.id}"
    assert set(data).isdisjoint({"video_path", "result_path", "debug_directory"})


def test_list_conversion_preserves_order_and_pagination(tmp_path: Path) -> None:
    first = _summary(tmp_path)
    second = _summary(tmp_path, verdict="flat")

    response = analysis_list_response([first, second], limit=20, offset=40)

    assert [item.id for item in response.items] == [first.id, second.id]
    assert response.limit == 20
    assert response.offset == 40


def test_public_schema_rejects_extra_fields_and_invalid_score(tmp_path: Path) -> None:
    data = analysis_response_from_stored(_stored_analysis(tmp_path)).model_dump()
    data["video_path"] = "C:/secret/video.mp4"
    data["score"] = 1.1

    with pytest.raises(ValidationError) as error:
        AnalysisResponse.model_validate(data)

    messages = str(error.value)
    assert "Extra inputs are not permitted" in messages
    assert "less than or equal to 1" in messages


def test_conversion_helpers_reject_wrong_input_types() -> None:
    with pytest.raises(TypeError, match="StoredAnalysis"):
        analysis_response_from_stored(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        analysis_response_from_result(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AnalysisSummary"):
        analysis_summary_response(object())  # type: ignore[arg-type]


def test_error_response_requires_nonempty_detail() -> None:
    assert ErrorResponse(detail="Not found").detail == "Not found"
    with pytest.raises(ValidationError):
        ErrorResponse(detail="")
