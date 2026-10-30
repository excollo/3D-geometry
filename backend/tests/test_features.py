"""Tests for central-ROI feature detection."""

import cv2
import numpy as np
import pytest
from unittest.mock import patch

from boxcheck.features import (
    FeatureDetectionError,
    create_central_roi_mask,
    create_detector,
    detect_features_for_frames,
    detect_frame_features,
    match_frame_features,
    match_selected_frame_pairs,
)
from boxcheck.types import FeatureData, FrameData, FramePair


def make_textured_frame(index: int = 0) -> FrameData:
    image = np.zeros((256, 256), dtype=np.uint8)
    for y in range(0, 256, 32):
        for x in range(0, 256, 32):
            if (x // 32 + y // 32) % 2 == 0:
                image[y : y + 32, x : x + 32] = 255
    original = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return FrameData(index, float(index), original, image)


def test_central_roi_mask_uses_requested_fractions() -> None:
    image = np.zeros((100, 200), dtype=np.uint8)

    mask = create_central_roi_mask(image, 0.80, 0.60)

    assert mask.shape == image.shape
    assert np.count_nonzero(mask) == 160 * 60
    assert mask[50, 100] == 255
    assert mask[0, 0] == 0


def test_sift_detects_features_inside_central_roi() -> None:
    frame = make_textured_frame()

    features = detect_frame_features(frame, "sift", 2000, 0.80, 0.80)

    assert len(features.keypoints) > 0
    assert features.descriptors is not None
    assert features.descriptors.shape[0] == len(features.keypoints)
    assert all(25 <= point.pt[0] <= 231 for point in features.keypoints)
    assert all(25 <= point.pt[1] <= 231 for point in features.keypoints)


@pytest.mark.parametrize("name", ["sift", "akaze", "orb"])
def test_supported_detectors_can_be_created(name: str) -> None:
    assert create_detector(name, 500) is not None


def test_unknown_detector_is_rejected() -> None:
    with pytest.raises(FeatureDetectionError, match="Unsupported"):
        create_detector("unknown", 500)


def test_lowe_ratio_test_keeps_clear_descriptor_matches() -> None:
    first = FeatureData(
        frame_index=0,
        keypoints=[cv2.KeyPoint(10, 20, 1), cv2.KeyPoint(30, 40, 1)],
        descriptors=np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
    )
    second = FeatureData(
        frame_index=1,
        keypoints=[
            cv2.KeyPoint(12, 20, 1),
            cv2.KeyPoint(32, 40, 1),
            cv2.KeyPoint(100, 100, 1),
        ],
        descriptors=np.array(
            [[0.0, 0.1], [10.1, 10.0], [100.0, 100.0]],
            dtype=np.float32,
        ),
    )

    matched = match_frame_features(first, second, lowe_ratio=0.75)

    assert len(matched.matches) == 2
    np.testing.assert_allclose(matched.first_points, [[10, 20], [30, 40]])
    np.testing.assert_allclose(matched.second_points, [[12, 20], [32, 40]])


def test_lowe_ratio_test_drops_ambiguous_match() -> None:
    first = FeatureData(
        frame_index=0,
        keypoints=[cv2.KeyPoint(10, 20, 1)],
        descriptors=np.array([[0.0, 0.0]], dtype=np.float32),
    )
    second = FeatureData(
        frame_index=1,
        keypoints=[cv2.KeyPoint(12, 20, 1), cv2.KeyPoint(13, 20, 1)],
        descriptors=np.array([[1.0, 0.0], [1.1, 0.0]], dtype=np.float32),
    )

    matched = match_frame_features(first, second, lowe_ratio=0.75)

    assert matched.matches == []
    assert matched.first_points.shape == (0, 2)
    assert matched.second_points.shape == (0, 2)


def test_matching_without_descriptors_returns_empty_result() -> None:
    first = FeatureData(frame_index=0)
    second = FeatureData(frame_index=1)

    matched = match_frame_features(first, second, lowe_ratio=0.75)

    assert len(matched.matches) == 0


def test_matching_rejects_invalid_lowe_ratio() -> None:
    first = FeatureData(frame_index=0)
    second = FeatureData(frame_index=1)

    with pytest.raises(ValueError, match="less than 1"):
        match_frame_features(first, second, lowe_ratio=1.0)


def test_batch_detection_skips_rejected_frames_and_caches_usable_frames() -> None:
    accepted = make_textured_frame(0)
    already_rejected = make_textured_frame(1)
    already_rejected.usable = False
    already_rejected.rejection_reason = "too_dark"

    with patch(
        "boxcheck.features.detect_frame_features",
        wraps=detect_frame_features,
    ) as detector_call:
        features_by_frame = detect_features_for_frames(
            [accepted, already_rejected],
            detector_name="sift",
            maximum_features=2000,
            minimum_features_per_frame=1,
            roi_width_fraction=0.80,
            roi_height_fraction=0.80,
        )

    assert set(features_by_frame) == {0}
    assert detector_call.call_count == 1


def test_batch_detection_marks_low_texture_frame() -> None:
    image = np.full((128, 128), 120, dtype=np.uint8)
    frame = FrameData(0, 0.0, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), image)

    features_by_frame = detect_features_for_frames(
        [frame], "sift", 2000, 40, 0.80, 0.80
    )

    assert features_by_frame == {}
    assert frame.usable is False
    assert frame.rejection_reason == "not_enough_texture"


def test_batch_matching_tracks_missing_and_too_small_pairs() -> None:
    first = FeatureData(
        0,
        [cv2.KeyPoint(float(index), 10, 1) for index in range(3)],
        np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32),
    )
    second = FeatureData(
        1,
        [cv2.KeyPoint(float(index + 2), 10, 1) for index in range(4)],
        np.array([[0, 0.1], [10.1, 10], [20, 20.1], [100, 100]], dtype=np.float32),
    )
    pairs = [FramePair(0, 1, 1.0), FramePair(1, 2, 1.0)]

    summary = match_selected_frame_pairs(
        {0: first, 1: second},
        pairs,
        lowe_ratio=0.75,
        minimum_matches_per_pair=4,
    )

    assert summary.pairs_requested == 2
    assert summary.pairs_accepted == 0
    assert summary.pairs_with_too_few_matches == 1
    assert summary.pairs_missing_features == 1


def test_real_sift_batch_integration_on_shifted_images() -> None:
    random = np.random.default_rng(7)
    first_image = np.zeros((320, 320), dtype=np.uint8)
    for x, y in random.integers(35, 285, size=(100, 2)):
        cv2.circle(first_image, (int(x), int(y)), 3, 255, -1)
    cv2.putText(
        first_image,
        "BOX 123",
        (70, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        180,
        2,
    )
    transform = np.float32([[1, 0, 8], [0, 1, 0]])
    second_image = cv2.warpAffine(first_image, transform, (320, 320))
    frames = [
        FrameData(0, 0.0, cv2.cvtColor(first_image, cv2.COLOR_GRAY2BGR), first_image),
        FrameData(1, 1.0, cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR), second_image),
    ]

    features_by_frame = detect_features_for_frames(
        frames,
        detector_name="sift",
        maximum_features=2000,
        minimum_features_per_frame=40,
        roi_width_fraction=0.90,
        roi_height_fraction=0.90,
    )
    summary = match_selected_frame_pairs(
        features_by_frame,
        [FramePair(0, 1, 1.0)],
        lowe_ratio=0.75,
        minimum_matches_per_pair=25,
    )

    assert summary.pairs_accepted == 1
    matches = summary.accepted_matches[0]
    movement = np.linalg.norm(matches.second_points - matches.first_points, axis=1)
    assert float(np.median(movement)) == pytest.approx(8.0, abs=1.0)
