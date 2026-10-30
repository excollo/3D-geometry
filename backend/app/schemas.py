"""Public API schemas and storage-to-response conversion helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.store import AnalysisSummary, StoredAnalysis


__all__ = [
    "AnalysisListResponse",
    "AnalysisMetricsResponse",
    "AnalysisResponse",
    "AnalysisSummaryResponse",
    "ErrorResponse",
    "PairMetricsResponse",
    "analysis_list_response",
    "analysis_response_from_result",
    "analysis_response_from_stored",
    "analysis_summary_response",
]


PublicVerdict = Literal["3d", "flat", "insufficient", "review"]
PublicLabel = Literal[
    "real",
    "flat_printout",
    "folded_printout",
    "screen",
    "bad_recording",
    "other",
]
AnalysisId = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
RelativeApiUrl = Annotated[str, Field(pattern=r"^/api/v1/")]


class ApiSchema(BaseModel):
    """Strict, immutable base for every public API schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PairMetricsResponse(ApiSchema):
    """Public geometry measurements for one frame pair."""

    first_frame_index: NonNegativeInt
    second_frame_index: NonNegativeInt
    time_gap_seconds: NonNegativeFloat
    match_count: NonNegativeInt
    median_movement_px: NonNegativeFloat
    homography_inlier_ratio: UnitFloat
    fundamental_inlier_ratio: UnitFloat
    homography_gric: NonNegativeFloat | None
    fundamental_gric: NonNegativeFloat | None
    median_parallax_px: NonNegativeFloat
    second_plane_found: bool
    second_plane_inliers: NonNegativeInt
    plane_normal_angle_degrees: NonNegativeFloat | None
    box_corner_found: bool
    pair_score: UnitFloat
    low_motion: bool
    low_parallax: bool
    pure_rotation_risk: bool


class AnalysisMetricsResponse(ApiSchema):
    """Public aggregate measurements for an entire video."""

    frames_sampled: NonNegativeInt
    frames_used: NonNegativeInt
    pairs_used: NonNegativeInt
    median_matches: NonNegativeFloat
    median_movement_px: NonNegativeFloat
    median_h_inlier_ratio: UnitFloat
    median_f_inlier_ratio: UnitFloat
    second_plane_pair_fraction: UnitFloat
    median_parallax_px: NonNegativeFloat
    median_brightness: NonNegativeFloat
    median_blur: NonNegativeFloat


class AnalysisResponse(ApiSchema):
    """Complete public representation of one saved analysis."""

    id: AnalysisId
    created_at: datetime
    verdict: PublicVerdict
    score: UnitFloat
    confidence: UnitFloat
    reason: Annotated[str, Field(min_length=1)]
    metrics: AnalysisMetricsResponse
    pairs: list[PairMetricsResponse]
    processing_ms: NonNegativeInt
    label: PublicLabel | None
    phone_model: str | None
    attempt: PositiveInt
    debug_frames: list[RelativeApiUrl]


class AnalysisSummaryResponse(ApiSchema):
    """Lightweight public history entry."""

    id: AnalysisId
    created_at: datetime
    label: PublicLabel | None
    phone_model: str | None
    attempt: PositiveInt
    verdict: PublicVerdict
    score: UnitFloat
    confidence: UnitFloat
    detail_url: RelativeApiUrl


class AnalysisListResponse(ApiSchema):
    """One page of analysis history."""

    items: list[AnalysisSummaryResponse]
    limit: Annotated[int, Field(ge=1, le=100)]
    offset: NonNegativeInt


class ErrorResponse(ApiSchema):
    """Stable JSON error body returned by application routes."""

    detail: Annotated[str, Field(min_length=1)]


def _detail_url(analysis_id: str) -> str:
    return f"/api/v1/analyses/{analysis_id}"


def _debug_urls(analysis_id: str, frame_count: int) -> list[str]:
    return [
        f"/api/v1/analyses/{analysis_id}/debug/{index}.jpg"
        for index in range(frame_count)
    ]


def analysis_response_from_stored(analysis: StoredAnalysis) -> AnalysisResponse:
    """Convert an in-memory stored analysis without exposing local paths."""
    if not isinstance(analysis, StoredAnalysis):
        raise TypeError("analysis must be a StoredAnalysis")
    return analysis_response_from_result(analysis.to_dict())


def analysis_response_from_result(result: Mapping[str, Any]) -> AnalysisResponse:
    """Convert one complete stored result dictionary to its public schema."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    analysis_id = result["id"]
    debug_frames = result["debug_frames"]
    if not isinstance(debug_frames, list):
        raise TypeError("debug_frames must be a list")

    return AnalysisResponse(
        id=analysis_id,
        created_at=result["created_at"],
        verdict=result["verdict"],
        score=result["score"],
        confidence=result["confidence"],
        reason=result["reason"],
        metrics=AnalysisMetricsResponse.model_validate(result["metrics"]),
        pairs=[PairMetricsResponse.model_validate(pair) for pair in result["pairs"]],
        processing_ms=result["processing_ms"],
        label=result["label"],
        phone_model=result["phone_model"],
        attempt=result["attempt"],
        debug_frames=_debug_urls(analysis_id, len(debug_frames)),
    )


def analysis_summary_response(summary: AnalysisSummary) -> AnalysisSummaryResponse:
    """Convert a SQLite-backed history entry without exposing local paths."""
    if not isinstance(summary, AnalysisSummary):
        raise TypeError("summary must be an AnalysisSummary")
    return AnalysisSummaryResponse(
        id=summary.id,
        created_at=summary.created_at,
        label=summary.label.value if summary.label is not None else None,
        phone_model=summary.phone_model,
        attempt=summary.attempt,
        verdict=summary.verdict,
        score=summary.score,
        confidence=summary.confidence,
        detail_url=_detail_url(summary.id),
    )


def analysis_list_response(
    summaries: Sequence[AnalysisSummary],
    *,
    limit: int,
    offset: int,
) -> AnalysisListResponse:
    """Convert one storage history page to its public list schema."""
    return AnalysisListResponse(
        items=[analysis_summary_response(summary) for summary in summaries],
        limit=limit,
        offset=offset,
    )
