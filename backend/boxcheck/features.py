"""Detect and match visual feature points between video frames."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from boxcheck.quality import apply_texture_gate
from boxcheck.types import (
    FeatureData,
    FeatureMatches,
    FrameData,
    FramePair,
    Image,
)


class FeatureDetectionError(ValueError):
    """Raised when feature detection settings or input are invalid."""


@dataclass(slots=True)
class FeatureMatchingSummary:
    """Batch result for matching a supplied list of selected frame pairs."""

    accepted_matches: list[FeatureMatches]
    pairs_requested: int
    pairs_accepted: int
    pairs_missing_features: int
    pairs_with_too_few_matches: int


def create_central_roi_mask(
    image: Image,
    width_fraction: float,
    height_fraction: float,
) -> Image:
    """Return a binary mask that keeps a centred rectangular image region."""
    if image.ndim != 2:
        raise FeatureDetectionError("Feature detection requires a grayscale image")
    for name, fraction in (
        ("width_fraction", width_fraction),
        ("height_fraction", height_fraction),
    ):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{name} must be greater than 0 and at most 1")

    height, width = image.shape
    roi_width = max(1, round(width * width_fraction))
    roi_height = max(1, round(height * height_fraction))
    left = (width - roi_width) // 2
    top = (height - roi_height) // 2

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top : top + roi_height, left : left + roi_width] = 255
    return mask


def create_detector(detector_name: str, maximum_features: int):
    """Create the configured OpenCV feature detector and descriptor."""
    if maximum_features < 1:
        raise ValueError("maximum_features must be at least 1")

    normalized_name = detector_name.strip().lower()
    if normalized_name == "sift":
        return cv2.SIFT_create(nfeatures=maximum_features)
    if normalized_name == "orb":
        return cv2.ORB_create(nfeatures=maximum_features)
    if normalized_name == "akaze":
        if hasattr(cv2, "AKAZE_create"):
            return cv2.AKAZE_create()
        if hasattr(cv2, "xfeatures2d_AKAZE"):
            detector = cv2.xfeatures2d_AKAZE.create()
            detector.setMaxPoints(maximum_features)
            return detector
        raise FeatureDetectionError("AKAZE is unavailable in this OpenCV build")
    raise FeatureDetectionError(
        f"Unsupported feature detector: {detector_name}. Use sift, akaze, or orb."
    )


def detect_frame_features(
    frame: FrameData,
    detector_name: str,
    maximum_features: int,
    roi_width_fraction: float,
    roi_height_fraction: float,
) -> FeatureData:
    """Detect keypoints and descriptors inside one frame's central ROI."""
    mask = create_central_roi_mask(
        frame.processed_image,
        roi_width_fraction,
        roi_height_fraction,
    )
    detector = create_detector(detector_name, maximum_features)
    keypoints, descriptors = detector.detectAndCompute(frame.processed_image, mask)

    return FeatureData(
        frame_index=frame.index,
        keypoints=list(keypoints),
        descriptors=descriptors,
    )


def detect_features_for_frames(
    frames: list[FrameData],
    detector_name: str,
    maximum_features: int,
    minimum_features_per_frame: int,
    roi_width_fraction: float,
    roi_height_fraction: float,
) -> dict[int, FeatureData]:
    """Detect once per usable frame and cache only texture-qualified results."""
    features_by_frame: dict[int, FeatureData] = {}

    for frame in frames:
        if not frame.usable:
            continue

        features = detect_frame_features(
            frame,
            detector_name,
            maximum_features,
            roi_width_fraction,
            roi_height_fraction,
        )
        if apply_texture_gate(frame, features, minimum_features_per_frame):
            features_by_frame[frame.index] = features

    return features_by_frame


def match_frame_features(
    first: FeatureData,
    second: FeatureData,
    lowe_ratio: float,
) -> FeatureMatches:
    """Match descriptors and remove ambiguous matches with Lowe's ratio test."""
    if first.frame_index == second.frame_index:
        raise ValueError("Feature matching requires two different frames")
    if not 0.0 < lowe_ratio < 1.0:
        raise ValueError("lowe_ratio must be greater than 0 and less than 1")

    if (
        first.descriptors is None
        or second.descriptors is None
        or len(first.descriptors) == 0
        or len(second.descriptors) < 2
    ):
        return _empty_matches(first.frame_index, second.frame_index)

    norm_type = (
        cv2.NORM_HAMMING
        if first.descriptors.dtype == np.uint8
        else cv2.NORM_L2
    )
    matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    candidate_groups = matcher.knnMatch(
        first.descriptors,
        second.descriptors,
        k=2,
    )

    accepted = []
    for candidates in candidate_groups:
        if len(candidates) < 2:
            continue
        best, second_best = candidates
        if best.distance < lowe_ratio * second_best.distance:
            accepted.append(best)

    first_points = np.asarray(
        [first.keypoints[match.queryIdx].pt for match in accepted],
        dtype=np.float32,
    ).reshape(-1, 2)
    second_points = np.asarray(
        [second.keypoints[match.trainIdx].pt for match in accepted],
        dtype=np.float32,
    ).reshape(-1, 2)

    return FeatureMatches(
        first_frame_index=first.frame_index,
        second_frame_index=second.frame_index,
        matches=accepted,
        first_points=first_points,
        second_points=second_points,
    )


def match_selected_frame_pairs(
    features_by_frame: dict[int, FeatureData],
    selected_pairs: list[FramePair],
    lowe_ratio: float,
    minimum_matches_per_pair: int,
) -> FeatureMatchingSummary:
    """Match selected pairs and retain only pairs with enough correspondences."""
    if minimum_matches_per_pair < 1:
        raise ValueError("minimum_matches_per_pair must be at least 1")

    accepted_matches: list[FeatureMatches] = []
    missing_features = 0
    too_few_matches = 0

    for pair in selected_pairs:
        first = features_by_frame.get(pair.first_frame_index)
        second = features_by_frame.get(pair.second_frame_index)
        if first is None or second is None:
            missing_features += 1
            continue

        matches = match_frame_features(first, second, lowe_ratio)
        if len(matches.matches) < minimum_matches_per_pair:
            too_few_matches += 1
            continue

        accepted_matches.append(matches)

    return FeatureMatchingSummary(
        accepted_matches=accepted_matches,
        pairs_requested=len(selected_pairs),
        pairs_accepted=len(accepted_matches),
        pairs_missing_features=missing_features,
        pairs_with_too_few_matches=too_few_matches,
    )


def _empty_matches(first_frame_index: int, second_frame_index: int) -> FeatureMatches:
    empty_points = np.empty((0, 2), dtype=np.float32)
    return FeatureMatches(
        first_frame_index=first_frame_index,
        second_frame_index=second_frame_index,
        matches=[],
        first_points=empty_points.copy(),
        second_points=empty_points.copy(),
    )
