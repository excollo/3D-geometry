"""Combine pair geometry into a video score, confidence, verdict, and reason."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxcheck.geometry import PairGeometryResult, SecondPlaneResult
from boxcheck.types import AnalysisMetrics, FramePair, PairMetrics, Verdict


@dataclass(frozen=True, slots=True)
class ScoringDecision:
    """Whole-video decision before API/storage metadata is attached."""

    verdict: Verdict
    score: float
    confidence: float
    reason: str
    metrics: AnalysisMetrics
    pairs: list[PairMetrics]


def score_geometry_pair(
    pair: FramePair,
    geometry: PairGeometryResult,
    second_plane: SecondPlaneResult,
    gric_weight: float,
    second_plane_weight: float,
    parallax_weight: float,
    minimum_parallax_px: float,
    minimum_parallax_points: int,
    minimum_parallax_point_fraction: float,
) -> PairMetrics:
    """Convert one pair's three geometry signals into a 0-to-1 score."""
    _validate_weights(gric_weight, second_plane_weight, parallax_weight)
    if minimum_parallax_px <= 0:
        raise ValueError("minimum_parallax_px must be greater than zero")
    if minimum_parallax_points < 1:
        raise ValueError("minimum_parallax_points must be at least 1")
    if not 0.0 <= minimum_parallax_point_fraction <= 1.0:
        raise ValueError("minimum_parallax_point_fraction must be between 0 and 1")
    if (
        pair.first_frame_index != geometry.first_frame_index
        or pair.second_frame_index != geometry.second_frame_index
    ):
        raise ValueError("FramePair and PairGeometryResult indices do not match")
    if geometry.homography_gric is None or geometry.fundamental_gric is None:
        raise ValueError("Both GRIC measurements are required to score a pair")

    gric_signal = float(np.clip(geometry.gric_3d_preference, 0.0, 1.0))
    second_plane_signal = 1.0 if second_plane.qualifies_as_box_surfaces else 0.0
    match_count = len(geometry.homography.inlier_mask)
    required_parallax_points = max(
        minimum_parallax_points,
        int(np.ceil(match_count * minimum_parallax_point_fraction)),
    )
    parallax_signal = (
        float(np.clip(geometry.parallax.median_px / minimum_parallax_px, 0.0, 1.0))
        if geometry.parallax.point_count >= required_parallax_points
        else 0.0
    )
    pair_score = float(
        gric_weight * gric_signal
        + second_plane_weight * second_plane_signal
        + parallax_weight * parallax_signal
    )

    return PairMetrics(
        first_frame_index=pair.first_frame_index,
        second_frame_index=pair.second_frame_index,
        time_gap_seconds=pair.time_gap_seconds,
        match_count=match_count,
        median_movement_px=geometry.median_movement_px,
        homography_inlier_ratio=geometry.homography.inlier_ratio,
        fundamental_inlier_ratio=geometry.fundamental.inlier_ratio,
        homography_gric=geometry.homography_gric,
        fundamental_gric=geometry.fundamental_gric,
        median_parallax_px=geometry.parallax.median_px,
        second_plane_found=second_plane.qualifies_as_box_surfaces,
        second_plane_inliers=second_plane.model.inlier_count,
        plane_normal_angle_degrees=(
            second_plane.normal_comparison.angle_degrees
            if second_plane.normal_comparison.success
            else None
        ),
        box_corner_found=second_plane.corner.found,
        pair_score=pair_score,
        low_motion=geometry.low_motion,
        low_parallax=geometry.low_parallax,
        pure_rotation_risk=geometry.pure_rotation_risk,
    )


def calculate_confidence(
    pair_scores: list[float],
    minimum_pairs_for_full_confidence: int,
    mad_scale: float,
) -> float:
    """Combine pair agreement and evidence quantity into 0-to-1 confidence."""
    if minimum_pairs_for_full_confidence < 1:
        raise ValueError("minimum_pairs_for_full_confidence must be at least 1")
    if mad_scale < 0:
        raise ValueError("mad_scale must be non-negative")
    if not pair_scores:
        return 0.0
    if any(not 0.0 <= score <= 1.0 for score in pair_scores):
        raise ValueError("All pair scores must be between 0 and 1")

    scores = np.asarray(pair_scores, dtype=np.float64)
    median_score = float(np.median(scores))
    median_absolute_deviation = float(np.median(np.abs(scores - median_score)))
    agreement = float(np.clip(1.0 - mad_scale * median_absolute_deviation, 0.0, 1.0))
    quantity = min(1.0, len(scores) / minimum_pairs_for_full_confidence)
    return float(np.clip(agreement * quantity, 0.0, 1.0))


def aggregate_pair_measurements(
    pairs: list[PairMetrics],
    frames_sampled: int,
    frames_used: int,
    median_brightness: float,
    median_blur: float,
    three_d_score_threshold: float,
    flat_score_threshold: float,
    minimum_confidence: float,
    minimum_median_movement_px: float,
    minimum_pairs_for_full_confidence: int,
    confidence_mad_scale: float,
) -> ScoringDecision:
    """Aggregate valid pair measurements and make a conservative verdict."""
    _validate_decision_thresholds(
        three_d_score_threshold,
        flat_score_threshold,
        minimum_confidence,
        minimum_median_movement_px,
    )
    scores = [pair.pair_score for pair in pairs]
    score = float(np.median(scores)) if scores else 0.5
    confidence = calculate_confidence(
        scores,
        minimum_pairs_for_full_confidence,
        confidence_mad_scale,
    )
    metrics = _build_metrics(
        pairs,
        frames_sampled,
        frames_used,
        median_brightness,
        median_blur,
    )
    verdict, reason = _choose_verdict(
        score,
        confidence,
        metrics,
        pairs,
        three_d_score_threshold,
        flat_score_threshold,
        minimum_confidence,
        minimum_median_movement_px,
    )
    return ScoringDecision(verdict, score, confidence, reason, metrics, pairs)


def _build_metrics(
    pairs: list[PairMetrics],
    frames_sampled: int,
    frames_used: int,
    median_brightness: float,
    median_blur: float,
) -> AnalysisMetrics:
    def median(attribute: str) -> float:
        return (
            float(np.median([getattr(pair, attribute) for pair in pairs]))
            if pairs
            else 0.0
        )

    second_plane_fraction = (
        sum(pair.second_plane_found for pair in pairs) / len(pairs) if pairs else 0.0
    )
    return AnalysisMetrics(
        frames_sampled=frames_sampled,
        frames_used=frames_used,
        pairs_used=len(pairs),
        median_matches=median("match_count"),
        median_movement_px=median("median_movement_px"),
        median_h_inlier_ratio=median("homography_inlier_ratio"),
        median_f_inlier_ratio=median("fundamental_inlier_ratio"),
        second_plane_pair_fraction=second_plane_fraction,
        median_parallax_px=median("median_parallax_px"),
        median_brightness=median_brightness,
        median_blur=median_blur,
    )


def _choose_verdict(
    score: float,
    confidence: float,
    metrics: AnalysisMetrics,
    pairs: list[PairMetrics],
    three_d_threshold: float,
    flat_threshold: float,
    minimum_confidence: float,
    minimum_movement: float,
) -> tuple[Verdict, str]:
    if not pairs:
        return Verdict.INSUFFICIENT, "Not enough valid frame pairs for geometry analysis."
    if metrics.median_movement_px < minimum_movement:
        return Verdict.INSUFFICIENT, "Not enough movement—move slowly around the box."
    if confidence < minimum_confidence:
        return (
            Verdict.INSUFFICIENT,
            "Frame-pair measurements disagree—record again with a slow, steady movement.",
        )

    pure_rotation_fraction = sum(pair.pure_rotation_risk for pair in pairs) / len(pairs)
    if score <= flat_threshold and pure_rotation_fraction >= 0.5:
        return (
            Verdict.INSUFFICIENT,
            "The camera mostly rotated in place—move sideways around the box.",
        )
    if score >= three_d_threshold:
        if metrics.second_plane_pair_fraction >= 0.5:
            reason = "Two angled surfaces with consistent parallax were detected."
        else:
            reason = "Consistent 3D motion and parallax were detected across frame pairs."
        return Verdict.THREE_D, reason
    if score <= flat_threshold and metrics.second_plane_pair_fraction == 0.0:
        return Verdict.FLAT, "One flat surface explains the movement across frame pairs."
    return (
        Verdict.INSUFFICIENT,
        "Geometry evidence is inconclusive—show the front, side, and top in one slow movement.",
    )


def _validate_weights(gric: float, second_plane: float, parallax: float) -> None:
    weights = (gric, second_plane, parallax)
    if any(weight < 0 for weight in weights):
        raise ValueError("Scoring weights must be non-negative")
    if not np.isclose(sum(weights), 1.0, atol=1e-9):
        raise ValueError("Scoring weights must sum to 1")


def _validate_decision_thresholds(
    three_d: float,
    flat: float,
    confidence: float,
    movement: float,
) -> None:
    if not 0 <= flat < three_d <= 1:
        raise ValueError("Score thresholds must satisfy 0 <= flat < 3d <= 1")
    if not 0 <= confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    if movement < 0:
        raise ValueError("minimum_median_movement_px must be non-negative")
