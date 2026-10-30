"""Tests for pair scoring, confidence, aggregation, and verdict rules."""

import numpy as np
import pytest

from boxcheck.geometry import (
    BoxCornerResult,
    GeometryModelResult,
    PairGeometryResult,
    ParallaxResult,
    PlaneNormalComparison,
    SecondPlaneResult,
)
from boxcheck.scoring import (
    aggregate_pair_measurements,
    calculate_confidence,
    score_geometry_pair,
)
from boxcheck.types import FramePair, PairMetrics, Verdict


def model(inlier_ratio: float = 0.8, count: int = 50) -> GeometryModelResult:
    inlier_count = round(inlier_ratio * count)
    mask = np.zeros(count, dtype=bool)
    mask[:inlier_count] = True
    return GeometryModelResult(
        success=True,
        matrix=np.eye(3),
        inlier_mask=mask,
        residuals_squared_px=np.ones(count),
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / count,
    )


def geometry_result(
    gric_signal: float = 0.8,
    parallax_px: float = 4.0,
    movement_px: float = 20.0,
    pure_rotation: bool = False,
) -> PairGeometryResult:
    homography_gric = 1000.0
    # Inverse of the normalized preference formula for a convenient valid pair.
    fundamental_gric = homography_gric * (1.0 - (gric_signal - 0.5) * 2.0)
    return PairGeometryResult(
        first_frame_index=0,
        second_frame_index=1,
        homography=model(0.6),
        fundamental=model(0.9),
        homography_gric=homography_gric,
        fundamental_gric=fundamental_gric,
        gric_3d_preference=gric_signal,
        parallax=ParallaxResult(parallax_px, parallax_px, 20),
        median_movement_px=movement_px,
        low_motion=movement_px < 8.0,
        low_parallax=parallax_px < 2.5,
        pure_rotation_risk=pure_rotation,
    )


def second_plane_result(qualifies: bool = True) -> SecondPlaneResult:
    normal = PlaneNormalComparison(
        True,
        45.0,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
        0.0,
        0.0,
    )
    corner = BoxCornerResult(qualifies, 5.0, np.array([100.0, 100.0]), 500.0, 500.0)
    return SecondPlaneResult(
        model_found=True,
        model=model(0.8),
        global_inlier_mask=np.ones(50, dtype=bool),
        normal_comparison=normal,
        orientation_distinct=qualifies,
        corner=corner,
        qualifies_as_box_surfaces=qualifies,
        reason="test",
    )


def pair_metric(
    score: float,
    index: int,
    second_plane: bool = False,
    movement: float = 20.0,
    pure_rotation: bool = False,
) -> PairMetrics:
    return PairMetrics(
        first_frame_index=index,
        second_frame_index=index + 1,
        time_gap_seconds=1.0,
        match_count=60,
        median_movement_px=movement,
        homography_inlier_ratio=0.65,
        fundamental_inlier_ratio=0.90,
        homography_gric=1000.0,
        fundamental_gric=700.0,
        median_parallax_px=5.0 if score > 0.5 else 1.0,
        second_plane_found=second_plane,
        second_plane_inliers=20 if second_plane else 0,
        plane_normal_angle_degrees=45.0 if second_plane else None,
        box_corner_found=second_plane,
        pair_score=score,
        pure_rotation_risk=pure_rotation,
    )


def aggregate(pairs: list[PairMetrics]):
    return aggregate_pair_measurements(
        pairs,
        frames_sampled=32,
        frames_used=27,
        median_brightness=124.0,
        median_blur=91.0,
        three_d_score_threshold=0.65,
        flat_score_threshold=0.35,
        minimum_confidence=0.60,
        minimum_median_movement_px=8.0,
        minimum_pairs_for_full_confidence=3,
        confidence_mad_scale=2.0,
    )


def test_pair_score_blends_three_configured_signals() -> None:
    result = score_geometry_pair(
        FramePair(0, 1, 1.0),
        geometry_result(gric_signal=0.8, parallax_px=3.0),
        second_plane_result(True),
        gric_weight=0.40,
        second_plane_weight=0.35,
        parallax_weight=0.25,
        minimum_parallax_px=2.5,
        minimum_parallax_points=12,
        minimum_parallax_point_fraction=0.15,
    )

    assert result.pair_score == pytest.approx(0.92)
    assert result.second_plane_found is True


def test_pair_score_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        score_geometry_pair(
            FramePair(0, 1, 1.0),
            geometry_result(),
            second_plane_result(),
            0.5,
            0.5,
            0.5,
            2.5,
            12,
            0.15,
        )


def test_parallax_needs_enough_supporting_points() -> None:
    geometry = geometry_result(gric_signal=0.2, parallax_px=20.0)
    geometry.parallax = ParallaxResult(20.0, 20.0, 2)

    result = score_geometry_pair(
        FramePair(0, 1, 1.0),
        geometry,
        second_plane_result(False),
        0.40,
        0.35,
        0.25,
        2.5,
        12,
        0.15,
    )

    assert result.pair_score == pytest.approx(0.08)


def test_confidence_is_high_for_three_consistent_pairs() -> None:
    confidence = calculate_confidence([0.79, 0.80, 0.81], 3, 2.0)

    assert confidence > 0.95


def test_confidence_is_low_when_pairs_disagree() -> None:
    confidence = calculate_confidence([0.0, 0.5, 1.0], 3, 2.0)

    assert confidence == pytest.approx(0.0)


def test_confidence_is_reduced_when_only_one_pair_exists() -> None:
    confidence = calculate_confidence([0.8], 3, 2.0)

    assert confidence == pytest.approx(1 / 3)


def test_consistent_high_scores_return_3d() -> None:
    decision = aggregate(
        [
            pair_metric(0.78, 0, True),
            pair_metric(0.82, 2, True),
            pair_metric(0.80, 4, True),
        ]
    )

    assert decision.verdict is Verdict.THREE_D
    assert decision.score == pytest.approx(0.80)
    assert decision.metrics.second_plane_pair_fraction == 1.0
    assert "Two angled surfaces" in decision.reason


def test_consistent_low_scores_without_second_plane_return_flat() -> None:
    decision = aggregate(
        [pair_metric(0.10, 0), pair_metric(0.12, 2), pair_metric(0.11, 4)]
    )

    assert decision.verdict is Verdict.FLAT
    assert "flat surface" in decision.reason


def test_low_motion_returns_insufficient_before_flat() -> None:
    decision = aggregate(
        [
            pair_metric(0.10, 0, movement=4.0),
            pair_metric(0.12, 2, movement=5.0),
            pair_metric(0.11, 4, movement=6.0),
        ]
    )

    assert decision.verdict is Verdict.INSUFFICIENT
    assert "movement" in decision.reason


def test_pure_rotation_risk_prevents_flat_verdict() -> None:
    decision = aggregate(
        [
            pair_metric(0.10, 0, pure_rotation=True),
            pair_metric(0.12, 2, pure_rotation=True),
            pair_metric(0.11, 4, pure_rotation=False),
        ]
    )

    assert decision.verdict is Verdict.INSUFFICIENT
    assert "rotated" in decision.reason


def test_middle_score_returns_inconclusive() -> None:
    decision = aggregate(
        [pair_metric(0.50, 0), pair_metric(0.52, 2), pair_metric(0.51, 4)]
    )

    assert decision.verdict is Verdict.INSUFFICIENT
    assert "inconclusive" in decision.reason


def test_no_pairs_return_insufficient_with_neutral_score() -> None:
    decision = aggregate([])

    assert decision.verdict is Verdict.INSUFFICIENT
    assert decision.score == 0.5
    assert decision.confidence == 0.0
    assert decision.metrics.pairs_used == 0


def test_metrics_are_aggregated_from_pair_medians() -> None:
    pairs = [pair_metric(0.7, 0, True), pair_metric(0.8, 2, False), pair_metric(0.9, 4, True)]

    decision = aggregate(pairs)

    assert decision.metrics.frames_sampled == 32
    assert decision.metrics.frames_used == 27
    assert decision.metrics.median_matches == 60
    assert decision.metrics.median_movement_px == 20
    assert decision.metrics.second_plane_pair_fraction == pytest.approx(2 / 3)
    assert decision.metrics.median_brightness == 124
    assert decision.metrics.median_blur == 91
