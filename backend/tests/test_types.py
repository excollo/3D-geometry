"""Tests for the shared checker data types."""

import json
from pathlib import Path

import numpy as np
import pytest

from boxcheck.types import (
    AnalysisInput,
    AnalysisMetrics,
    AnalysisResult,
    FrameData,
    FeatureMatches,
    FramePair,
    GroundTruthLabel,
    PairMetrics,
    Verdict,
)


def test_verdict_has_only_part_one_outcomes() -> None:
    assert {verdict.value for verdict in Verdict} == {
        "3d",
        "flat",
        "insufficient",
    }


def test_analysis_input_converts_strings_to_typed_values() -> None:
    analysis_input = AnalysisInput(
        video_path="sample.mp4",
        label="real",
        phone_model="Pixel 7a",
    )

    assert analysis_input.video_path == Path("sample.mp4")
    assert analysis_input.label is GroundTruthLabel.REAL


def test_frame_data_accepts_uint8_images() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    frame = FrameData(
        index=0,
        timestamp_seconds=0.0,
        original_image=image,
        processed_image=image.copy(),
    )

    assert frame.usable is True


def test_pair_rejects_an_out_of_range_score() -> None:
    with pytest.raises(ValueError, match="pair_score must be between 0 and 1"):
        PairMetrics(0, 1, 1.0, pair_score=1.1)


def test_metrics_reject_more_used_frames_than_sampled() -> None:
    with pytest.raises(ValueError, match="frames_used cannot exceed"):
        AnalysisMetrics(frames_sampled=2, frames_used=3)


@pytest.mark.parametrize("field,value", [("score", -0.1), ("confidence", 1.1)])
def test_result_rejects_values_outside_zero_to_one(field: str, value: float) -> None:
    values = {
        "verdict": Verdict.INSUFFICIENT,
        "score": 0.0,
        "confidence": 0.5,
        "reason": "Not enough movement.",
    }
    values[field] = value

    with pytest.raises(ValueError, match=f"{field} must be between 0 and 1"):
        AnalysisResult(**values)


def test_result_converts_to_json_compatible_dictionary() -> None:
    result = AnalysisResult(
        verdict=Verdict.THREE_D,
        score=0.81,
        confidence=0.74,
        reason="Two surfaces at different angles were detected.",
        metrics=AnalysisMetrics(frames_sampled=32, frames_used=27),
        pairs=[PairMetrics(0, 4, 1.0, match_count=86, pair_score=0.82)],
        processing_ms=2800,
        label=GroundTruthLabel.REAL,
        phone_model="Pixel 7a",
        debug_frames=[Path("debug/0.jpg")],
    )

    data = result.to_dict()

    assert data["verdict"] == "3d"
    assert data["label"] == "real"
    assert data["metrics"]["frames_used"] == 27
    assert data["pairs"][0]["match_count"] == 86
    assert data["debug_frames"] == [str(Path("debug/0.jpg"))]
    json.dumps(data)


def test_optional_input_values_can_be_none() -> None:
    analysis_input = AnalysisInput(video_path=Path("sample.mp4"))

    assert analysis_input.label is None
    assert analysis_input.phone_model is None


def test_feature_matches_require_aligned_point_arrays() -> None:
    with pytest.raises(ValueError, match="first_points must have shape"):
        FeatureMatches(
            first_frame_index=0,
            second_frame_index=1,
            matches=[object()],
            first_points=np.empty((0, 2), dtype=np.float32),
            second_points=np.empty((1, 2), dtype=np.float32),
        )


def test_frame_pair_requires_different_frames_and_positive_gap() -> None:
    with pytest.raises(ValueError, match="different frames"):
        FramePair(0, 0, 1.0)
    with pytest.raises(ValueError, match="greater than zero"):
        FramePair(0, 1, 0.0)
