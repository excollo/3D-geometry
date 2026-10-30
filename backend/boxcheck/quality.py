"""Measure frame quality, texture, and camera movement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from boxcheck.preprocessing import convert_to_grayscale
from boxcheck.types import FeatureData, FeatureMatches, FrameData, Image


TOO_DARK_REASON = "too_dark"
TOO_BRIGHT_REASON = "too_bright"
TOO_BLURRY_REASON = "too_blurry"
NOT_ENOUGH_FRAMES_REASON = "not_enough_usable_frames"
NOT_ENOUGH_TEXTURE_REASON = "not_enough_texture"
NOT_ENOUGH_MATCHES_REASON = "not_enough_matches"
NOT_ENOUGH_MOVEMENT_REASON = "not_enough_movement"


@dataclass(frozen=True, slots=True)
class FrameQualitySummary:
    """Whole-video outcome after per-frame quality gates."""

    passed: bool
    frames_sampled: int
    frames_usable: int
    usable_fraction: float
    rejection_counts: dict[str, int]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class MovementQualitySummary:
    """Whole-video camera-movement result from valid matched frame pairs."""

    passed: bool
    median_movement_px: float
    pair_movements_px: list[float]
    pairs_used: int
    matches_used: int
    reason: str | None = None


class QualityMeasurementError(ValueError):
    """Raised when an image is unsuitable for a quality measurement."""


def calculate_brightness(grayscale_image: Image) -> float:
    """Return the mean grayscale brightness on the 0-to-255 pixel scale."""
    if grayscale_image.ndim != 2:
        raise QualityMeasurementError(
            "Brightness measurement requires a grayscale image"
        )
    if grayscale_image.size == 0:
        raise QualityMeasurementError("Cannot measure brightness of an empty image")

    return float(np.mean(grayscale_image))


def measure_frame_brightness(frame: FrameData) -> float:
    """Measure original capture brightness before contrast enhancement."""
    original_grayscale = convert_to_grayscale(frame.original_image)
    brightness = calculate_brightness(original_grayscale)
    frame.brightness = brightness
    return brightness


def measure_brightness_for_frames(frames: list[FrameData]) -> list[float]:
    """Measure brightness for every sampled frame in order."""
    return [measure_frame_brightness(frame) for frame in frames]


def calculate_blur_score(grayscale_image: Image) -> float:
    """Return Laplacian variance; larger values indicate sharper edges."""
    if grayscale_image.ndim != 2:
        raise QualityMeasurementError("Blur measurement requires a grayscale image")
    if grayscale_image.size == 0:
        raise QualityMeasurementError("Cannot measure blur of an empty image")

    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    return float(laplacian.var())


def measure_frame_blur(frame: FrameData) -> float:
    """Measure sharpness on one processed frame and store the score."""
    blur_score = calculate_blur_score(frame.processed_image)
    frame.blur_score = blur_score
    return blur_score


def measure_blur_for_frames(frames: list[FrameData]) -> list[float]:
    """Measure blur scores for every sampled frame in order."""
    return [measure_frame_blur(frame) for frame in frames]


def apply_blur_gate(frame: FrameData, minimum_blur_score: float) -> bool:
    """Measure one frame and reject it when its sharpness is too low."""
    if minimum_blur_score < 0:
        raise ValueError("minimum_blur_score must be non-negative")

    blur_score = measure_frame_blur(frame)
    if frame.usable and blur_score < minimum_blur_score:
        frame.usable = False
        frame.rejection_reason = TOO_BLURRY_REASON

    return frame.usable


def apply_blur_gates(
    frames: list[FrameData],
    minimum_blur_score: float,
) -> list[FrameData]:
    """Apply the blur gate and return only frames that remain usable."""
    if minimum_blur_score < 0:
        raise ValueError("minimum_blur_score must be non-negative")

    for frame in frames:
        apply_blur_gate(frame, minimum_blur_score)
    return [frame for frame in frames if frame.usable]


def apply_texture_gate(
    frame: FrameData,
    features: FeatureData,
    minimum_feature_count: int,
) -> bool:
    """Reject a frame when too few central-ROI feature points were detected."""
    if minimum_feature_count < 1:
        raise ValueError("minimum_feature_count must be at least 1")
    if features.frame_index != frame.index:
        raise ValueError("Feature data does not belong to the supplied frame")

    if frame.usable and len(features.keypoints) < minimum_feature_count:
        frame.usable = False
        frame.rejection_reason = NOT_ENOUGH_TEXTURE_REASON

    return frame.usable


def calculate_median_match_movement(matches: FeatureMatches) -> float:
    """Return the median Euclidean displacement of matched feature points."""
    if not matches.matches:
        return 0.0

    displacements = np.linalg.norm(
        matches.second_points - matches.first_points,
        axis=1,
    )
    return float(np.median(displacements))


def evaluate_camera_movement(
    matched_pairs: list[FeatureMatches],
    minimum_matches_per_pair: int,
    minimum_median_movement_px: float,
) -> MovementQualitySummary:
    """Require enough reliable matches and camera translation across the video."""
    if minimum_matches_per_pair < 1:
        raise ValueError("minimum_matches_per_pair must be at least 1")
    if minimum_median_movement_px < 0:
        raise ValueError("minimum_median_movement_px must be non-negative")

    valid_pairs = [
        matches
        for matches in matched_pairs
        if len(matches.matches) >= minimum_matches_per_pair
    ]
    if not valid_pairs:
        return MovementQualitySummary(
            passed=False,
            median_movement_px=0.0,
            pair_movements_px=[],
            pairs_used=0,
            matches_used=0,
            reason=NOT_ENOUGH_MATCHES_REASON,
        )

    pair_movements = [
        calculate_median_match_movement(matches) for matches in valid_pairs
    ]
    video_median_movement = float(np.median(pair_movements))
    passed = video_median_movement >= minimum_median_movement_px

    return MovementQualitySummary(
        passed=passed,
        median_movement_px=video_median_movement,
        pair_movements_px=pair_movements,
        pairs_used=len(valid_pairs),
        matches_used=sum(len(matches.matches) for matches in valid_pairs),
        reason=None if passed else NOT_ENOUGH_MOVEMENT_REASON,
    )


def apply_brightness_gate(
    frame: FrameData,
    minimum_brightness: float,
    maximum_brightness: float,
) -> bool:
    """Measure one frame and mark it unusable if exposure is unacceptable."""
    _validate_brightness_thresholds(minimum_brightness, maximum_brightness)
    brightness = measure_frame_brightness(frame)

    if brightness < minimum_brightness:
        frame.usable = False
        frame.rejection_reason = TOO_DARK_REASON
    elif brightness > maximum_brightness:
        frame.usable = False
        frame.rejection_reason = TOO_BRIGHT_REASON

    return frame.usable


def apply_brightness_gates(
    frames: list[FrameData],
    minimum_brightness: float,
    maximum_brightness: float,
) -> list[FrameData]:
    """Apply the exposure gate and return only currently usable frames."""
    _validate_brightness_thresholds(minimum_brightness, maximum_brightness)
    for frame in frames:
        apply_brightness_gate(frame, minimum_brightness, maximum_brightness)
    return [frame for frame in frames if frame.usable]


def _validate_brightness_thresholds(
    minimum_brightness: float,
    maximum_brightness: float,
) -> None:
    if not 0 <= minimum_brightness <= 255:
        raise ValueError("minimum_brightness must be between 0 and 255")
    if not 0 <= maximum_brightness <= 255:
        raise ValueError("maximum_brightness must be between 0 and 255")
    if minimum_brightness > maximum_brightness:
        raise ValueError("minimum_brightness cannot exceed maximum_brightness")


def summarize_usable_frames(
    frames: list[FrameData],
    minimum_usable_frame_fraction: float,
    minimum_usable_frames: int,
) -> FrameQualitySummary:
    """Decide whether enough frames survived the brightness and blur gates."""
    if not 0.0 <= minimum_usable_frame_fraction <= 1.0:
        raise ValueError("minimum_usable_frame_fraction must be between 0 and 1")
    if minimum_usable_frames < 1:
        raise ValueError("minimum_usable_frames must be at least 1")

    frames_sampled = len(frames)
    frames_usable = sum(frame.usable for frame in frames)
    usable_fraction = frames_usable / frames_sampled if frames_sampled else 0.0
    rejection_counts = dict(
        Counter(
            frame.rejection_reason or NOT_ENOUGH_FRAMES_REASON
            for frame in frames
            if not frame.usable
        )
    )

    passed = (
        frames_usable >= minimum_usable_frames
        and usable_fraction >= minimum_usable_frame_fraction
    )
    reason = None if passed else _dominant_quality_failure(rejection_counts)

    return FrameQualitySummary(
        passed=passed,
        frames_sampled=frames_sampled,
        frames_usable=frames_usable,
        usable_fraction=usable_fraction,
        rejection_counts=rejection_counts,
        reason=reason,
    )


def _dominant_quality_failure(rejection_counts: dict[str, int]) -> str:
    if not rejection_counts:
        return NOT_ENOUGH_FRAMES_REASON

    priority = {
        TOO_DARK_REASON: 0,
        TOO_BRIGHT_REASON: 1,
        TOO_BLURRY_REASON: 2,
        NOT_ENOUGH_TEXTURE_REASON: 3,
        NOT_ENOUGH_FRAMES_REASON: 4,
    }
    return min(
        rejection_counts,
        key=lambda reason: (-rejection_counts[reason], priority.get(reason, 99)),
    )
