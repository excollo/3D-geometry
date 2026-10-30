"""Coordinate the complete standalone video analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from boxcheck.debug_draw import build_debug_pair_input, save_debug_pair_images
from boxcheck.features import detect_features_for_frames, match_selected_frame_pairs
from boxcheck.geometry import (
    analyze_pair_geometry,
    analyze_second_plane,
    build_camera_matrix,
)
from boxcheck.pairing import select_frame_pairs
from boxcheck.quality import (
    NOT_ENOUGH_MATCHES_REASON,
    NOT_ENOUGH_MOVEMENT_REASON,
    TOO_BLURRY_REASON,
    TOO_BRIGHT_REASON,
    TOO_DARK_REASON,
    apply_blur_gates,
    apply_brightness_gates,
    evaluate_camera_movement,
    summarize_usable_frames,
)
from boxcheck.scoring import aggregate_pair_measurements, score_geometry_pair
from boxcheck.settings import load_settings
from boxcheck.types import (
    AnalysisMetrics,
    AnalysisResult,
    FrameData,
    GroundTruthLabel,
    Verdict,
)
from boxcheck.video import VideoValidationError, extract_sampled_frames


QUALITY_REASONS = {
    TOO_DARK_REASON: "Too dark—record in better lighting.",
    TOO_BRIGHT_REASON: "Too bright or overexposed—avoid glare and direct light.",
    TOO_BLURRY_REASON: "Too blurry—move more slowly and keep the box in focus.",
    "not_enough_texture": "Not enough detail—include the printed side and label.",
    "not_enough_usable_frames": "Not enough usable frames—record the box for longer.",
    NOT_ENOUGH_MATCHES_REASON: "Not enough reliable matches—keep the box centred and visible.",
    NOT_ENOUGH_MOVEMENT_REASON: "Not enough movement—move slowly around the box.",
}


def analyze_video(
    video_path: str | Path,
    label: GroundTruthLabel | str | None = None,
    phone_model: str | None = None,
    thresholds_path: str | Path | None = None,
    phones_path: str | Path | None = None,
    debug_output_directory: str | Path | None = None,
) -> AnalysisResult:
    """Analyze one video without depending on FastAPI or storage code."""
    started = perf_counter()
    settings = load_settings(thresholds_path, phones_path)
    thresholds = settings["thresholds"]
    phone_settings = settings["phones"]
    video_settings = thresholds["video"]
    preprocessing = thresholds["preprocessing"]
    quality = thresholds["quality"]
    roi = thresholds["roi"]
    features = thresholds["features"]
    pairing = thresholds["pairing"]
    geometry_settings = thresholds["geometry"]
    scoring = thresholds["scoring"]
    debug = thresholds["debug"]
    typed_label = GroundTruthLabel(label) if isinstance(label, str) else label

    try:
        metadata, frames = extract_sampled_frames(
            video_path,
            video_settings["maximum_duration_seconds"],
            video_settings["sample_frames_per_second"],
            video_settings["maximum_working_long_side_px"],
            preprocessing["use_clahe"],
            preprocessing["clahe_clip_limit"],
            preprocessing["clahe_grid_size"],
        )
    except VideoValidationError as error:
        return _insufficient_result(
            str(error), started, typed_label, phone_model, AnalysisMetrics()
        )

    apply_brightness_gates(
        frames,
        quality["minimum_brightness"],
        quality["maximum_brightness"],
    )
    apply_blur_gates(frames, quality["minimum_laplacian_variance"])
    features_by_frame = detect_features_for_frames(
        frames,
        features["detector"],
        features["maximum_features"],
        features["minimum_features_per_frame"],
        roi["width_fraction"],
        roi["height_fraction"],
    )
    frame_quality = summarize_usable_frames(
        frames,
        quality["minimum_usable_frame_fraction"],
        video_settings["minimum_sampled_frames"],
    )
    base_metrics = _base_metrics(frames, frame_quality.frames_usable)
    if not frame_quality.passed:
        reason = QUALITY_REASONS.get(
            frame_quality.reason,
            "Not enough usable frames—record again with better lighting and focus.",
        )
        return _insufficient_result(reason, started, typed_label, phone_model, base_metrics)

    selected_pairs = select_frame_pairs(
        frames,
        pairing["minimum_gap_seconds"],
        pairing["maximum_gap_seconds"],
        pairing["maximum_pairs"],
    )
    if not selected_pairs:
        return _insufficient_result(
            "No useful frame pairs—record a longer, steady move around the box.",
            started,
            typed_label,
            phone_model,
            base_metrics,
        )

    matching = match_selected_frame_pairs(
        features_by_frame,
        selected_pairs,
        features["lowe_ratio"],
        features["minimum_matches_per_pair"],
    )
    movement = evaluate_camera_movement(
        matching.accepted_matches,
        features["minimum_matches_per_pair"],
        quality["minimum_median_movement_px"],
    )
    if not movement.passed:
        reason = QUALITY_REASONS.get(
            movement.reason,
            "Insufficient camera movement or feature matches.",
        )
        return _insufficient_result(reason, started, typed_label, phone_model, base_metrics)

    frames_by_index = {frame.index: frame for frame in frames}
    pairs_by_indices = {
        (pair.first_frame_index, pair.second_frame_index): pair
        for pair in selected_pairs
    }
    field_of_view = _resolve_horizontal_fov(phone_model, phone_settings)
    scored_pairs = []
    debug_pairs = []

    for matches in matching.accepted_matches:
        pair = pairs_by_indices[(matches.first_frame_index, matches.second_frame_index)]
        first_frame = frames_by_index[matches.first_frame_index]
        second_frame = frames_by_index[matches.second_frame_index]
        image_height, image_width = first_frame.processed_image.shape[:2]
        camera_matrix = build_camera_matrix(image_width, image_height, field_of_view)
        pair_geometry = analyze_pair_geometry(
            matches,
            geometry_settings["ransac_reprojection_threshold_px"],
            geometry_settings["ransac_confidence"],
            geometry_settings["ransac_max_iterations"],
            geometry_settings["minimum_homography_inliers"],
            geometry_settings["minimum_fundamental_inliers"],
            geometry_settings["minimum_parallax_px"],
            quality["minimum_median_movement_px"],
        )
        if (
            pair_geometry.homography_gric is None
            or pair_geometry.fundamental_gric is None
        ):
            continue
        second_plane = analyze_second_plane(
            matches,
            pair_geometry.homography,
            camera_matrix,
            image_width,
            image_height,
            roi["width_fraction"],
            roi["height_fraction"],
            geometry_settings["ransac_reprojection_threshold_px"],
            geometry_settings["ransac_confidence"],
            geometry_settings["ransac_max_iterations"],
            geometry_settings["minimum_second_plane_inliers"],
            geometry_settings["minimum_second_plane_normal_angle_degrees"],
            geometry_settings["maximum_decomposition_rotation_difference"],
            geometry_settings["maximum_decomposition_translation_direction_difference"],
            geometry_settings["box_corner_edge_tolerance_px"],
            geometry_settings["minimum_plane_hull_area_px2"],
        )
        pair_metrics = score_geometry_pair(
            pair,
            pair_geometry,
            second_plane,
            scoring["gric_weight"],
            scoring["second_plane_weight"],
            scoring["parallax_weight"],
            geometry_settings["minimum_parallax_px"],
            geometry_settings["minimum_parallax_points"],
            geometry_settings["minimum_parallax_point_fraction"],
        )
        scored_pairs.append(pair_metrics)
        debug_pairs.append(
            build_debug_pair_input(
                first_frame,
                second_frame,
                matches,
                pair_geometry,
                second_plane,
                pair_metrics,
            )
        )

    decision = aggregate_pair_measurements(
        scored_pairs,
        frames_sampled=len(frames),
        frames_used=frame_quality.frames_usable,
        median_brightness=base_metrics.median_brightness,
        median_blur=base_metrics.median_blur,
        three_d_score_threshold=scoring["three_d_score_threshold"],
        flat_score_threshold=scoring["flat_score_threshold"],
        minimum_confidence=scoring["minimum_confidence"],
        minimum_median_movement_px=quality["minimum_median_movement_px"],
        minimum_pairs_for_full_confidence=scoring[
            "minimum_pairs_for_full_confidence"
        ],
        confidence_mad_scale=scoring["confidence_mad_scale"],
    )

    debug_paths: list[Path] = []
    if debug["enabled"] and debug_output_directory is not None:
        debug_paths = save_debug_pair_images(
            debug_output_directory,
            debug_pairs,
            debug["maximum_annotated_pairs"],
        )

    return AnalysisResult(
        verdict=decision.verdict,
        score=decision.score,
        confidence=decision.confidence,
        reason=decision.reason,
        metrics=decision.metrics,
        pairs=decision.pairs,
        processing_ms=_elapsed_ms(started),
        label=typed_label,
        phone_model=phone_model,
        debug_frames=debug_paths,
    )


def _base_metrics(frames: list[FrameData], frames_used: int) -> AnalysisMetrics:
    brightness = [frame.brightness for frame in frames]
    blur = [frame.blur_score for frame in frames]
    return AnalysisMetrics(
        frames_sampled=len(frames),
        frames_used=frames_used,
        median_brightness=float(np.median(brightness)) if brightness else 0.0,
        median_blur=float(np.median(blur)) if blur else 0.0,
    )


def _resolve_horizontal_fov(phone_model: str | None, phones: dict) -> float:
    if phone_model:
        phone = phones["phones"].get(phone_model)
        if phone and "horizontal_field_of_view_degrees" in phone:
            return float(phone["horizontal_field_of_view_degrees"])
    return float(phones["default"]["horizontal_field_of_view_degrees"])


def _insufficient_result(
    reason: str,
    started: float,
    label: GroundTruthLabel | None,
    phone_model: str | None,
    metrics: AnalysisMetrics,
) -> AnalysisResult:
    return AnalysisResult(
        verdict=Verdict.INSUFFICIENT,
        score=0.5,
        confidence=0.0,
        reason=reason,
        metrics=metrics,
        processing_ms=_elapsed_ms(started),
        label=label,
        phone_model=phone_model,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))
