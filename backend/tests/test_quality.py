"""Tests for frame quality measurements and gates."""

import cv2
import numpy as np
import pytest

from boxcheck.quality import (
    QualityMeasurementError,
    TOO_BRIGHT_REASON,
    TOO_BLURRY_REASON,
    TOO_DARK_REASON,
    NOT_ENOUGH_FRAMES_REASON,
    NOT_ENOUGH_MATCHES_REASON,
    NOT_ENOUGH_MOVEMENT_REASON,
    NOT_ENOUGH_TEXTURE_REASON,
    apply_blur_gate,
    apply_blur_gates,
    apply_texture_gate,
    calculate_median_match_movement,
    apply_brightness_gate,
    apply_brightness_gates,
    calculate_blur_score,
    calculate_brightness,
    measure_blur_for_frames,
    measure_brightness_for_frames,
    measure_frame_blur,
    measure_frame_brightness,
    evaluate_camera_movement,
    summarize_usable_frames,
)
from boxcheck.types import FeatureData, FeatureMatches, FrameData


def make_frame(index: int, brightness: int) -> FrameData:
    original = np.full((4, 6, 3), brightness, dtype=np.uint8)
    processed = np.full((4, 6), brightness, dtype=np.uint8)
    return FrameData(
        index=index,
        timestamp_seconds=float(index),
        original_image=original,
        processed_image=processed,
    )


@pytest.mark.parametrize(
    ("pixel_value", "expected_brightness"),
    [(0, 0.0), (127, 127.0), (255, 255.0)],
)
def test_calculate_brightness_uses_zero_to_255_scale(
    pixel_value: int,
    expected_brightness: float,
) -> None:
    image = np.full((8, 8), pixel_value, dtype=np.uint8)

    assert calculate_brightness(image) == expected_brightness


def test_calculate_brightness_returns_mean_pixel_value() -> None:
    image = np.array([[0, 100], [200, 100]], dtype=np.uint8)

    assert calculate_brightness(image) == 100.0


def test_measure_frame_brightness_stores_value_on_frame() -> None:
    frame = make_frame(0, 140)

    measured = measure_frame_brightness(frame)

    assert measured == 140.0
    assert frame.brightness == 140.0


def test_measure_brightness_for_frames_preserves_order() -> None:
    frames = [make_frame(0, 40), make_frame(1, 120), make_frame(2, 220)]

    values = measure_brightness_for_frames(frames)

    assert values == [40.0, 120.0, 220.0]
    assert [frame.brightness for frame in frames] == values


def test_brightness_rejects_colour_input() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(QualityMeasurementError, match="grayscale"):
        calculate_brightness(image)


@pytest.mark.parametrize("brightness", [35, 120, 225])
def test_brightness_gate_accepts_values_inside_inclusive_limits(
    brightness: int,
) -> None:
    frame = make_frame(0, brightness)

    usable = apply_brightness_gate(frame, 35.0, 225.0)

    assert usable is True
    assert frame.usable is True
    assert frame.rejection_reason is None


def test_brightness_gate_marks_dark_frame() -> None:
    frame = make_frame(0, 34)

    usable = apply_brightness_gate(frame, 35.0, 225.0)

    assert usable is False
    assert frame.rejection_reason == TOO_DARK_REASON


def test_brightness_gate_marks_bright_frame() -> None:
    frame = make_frame(0, 226)

    usable = apply_brightness_gate(frame, 35.0, 225.0)

    assert usable is False
    assert frame.rejection_reason == TOO_BRIGHT_REASON


def test_brightness_gate_uses_original_image_before_clahe() -> None:
    frame = make_frame(0, 20)
    frame.processed_image[:] = 150

    apply_brightness_gate(frame, 35.0, 225.0)

    assert frame.brightness == 20.0
    assert frame.rejection_reason == TOO_DARK_REASON


def test_batch_brightness_gate_returns_only_usable_frames() -> None:
    frames = [make_frame(0, 20), make_frame(1, 120), make_frame(2, 240)]

    usable_frames = apply_brightness_gates(frames, 35.0, 225.0)

    assert [frame.index for frame in usable_frames] == [1]
    assert [frame.rejection_reason for frame in frames] == [
        TOO_DARK_REASON,
        None,
        TOO_BRIGHT_REASON,
    ]


def test_brightness_gate_rejects_reversed_thresholds() -> None:
    frame = make_frame(0, 120)

    with pytest.raises(ValueError, match="cannot exceed"):
        apply_brightness_gate(frame, 225.0, 35.0)


def test_uniform_image_has_zero_blur_score() -> None:
    image = np.full((32, 32), 120, dtype=np.uint8)

    assert calculate_blur_score(image) == 0.0


def test_sharp_edges_have_higher_blur_score_than_smoothed_edges() -> None:
    sharp = np.zeros((64, 64), dtype=np.uint8)
    sharp[:, 32:] = 255
    smoothed = cv2.GaussianBlur(sharp, (15, 15), 0)

    sharp_score = calculate_blur_score(sharp)
    smoothed_score = calculate_blur_score(smoothed)

    assert sharp_score > smoothed_score


def test_measure_frame_blur_stores_score_on_frame() -> None:
    frame = make_frame(0, 120)
    frame.processed_image[:, 3:] = 255

    measured = measure_frame_blur(frame)

    assert measured > 0
    assert frame.blur_score == measured


def test_measure_blur_for_frames_preserves_order() -> None:
    smooth = make_frame(0, 120)
    sharp = make_frame(1, 0)
    sharp.processed_image[:, 3:] = 255

    scores = measure_blur_for_frames([smooth, sharp])

    assert scores[0] == 0.0
    assert scores[1] > scores[0]


def test_blur_measurement_rejects_colour_input() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(QualityMeasurementError, match="grayscale"):
        calculate_blur_score(image)


def test_blur_gate_marks_low_scoring_frame_as_unusable() -> None:
    frame = make_frame(0, 120)

    usable = apply_blur_gate(frame, minimum_blur_score=60.0)

    assert usable is False
    assert frame.blur_score == 0.0
    assert frame.rejection_reason == TOO_BLURRY_REASON


def test_blur_gate_accepts_score_at_exact_threshold() -> None:
    frame = make_frame(0, 120)
    measured_score = calculate_blur_score(frame.processed_image)

    usable = apply_blur_gate(frame, minimum_blur_score=measured_score)

    assert usable is True
    assert frame.rejection_reason is None


def test_blur_gate_keeps_an_earlier_rejection_reason() -> None:
    frame = make_frame(0, 20)
    apply_brightness_gate(frame, 35.0, 225.0)

    apply_blur_gate(frame, minimum_blur_score=60.0)

    assert frame.usable is False
    assert frame.rejection_reason == TOO_DARK_REASON
    assert frame.blur_score == 0.0


def test_batch_blur_gate_returns_only_remaining_usable_frames() -> None:
    blurry = make_frame(0, 120)
    sharp = make_frame(1, 0)
    sharp.processed_image[:, 3:] = 255

    sharp_score = calculate_blur_score(sharp.processed_image)
    usable_frames = apply_blur_gates(
        [blurry, sharp],
        minimum_blur_score=sharp_score / 2,
    )

    assert [frame.index for frame in usable_frames] == [1]
    assert blurry.rejection_reason == TOO_BLURRY_REASON


def test_blur_gate_rejects_negative_threshold() -> None:
    frame = make_frame(0, 120)

    with pytest.raises(ValueError, match="non-negative"):
        apply_blur_gate(frame, minimum_blur_score=-1.0)


def test_usable_frame_summary_passes_at_exact_limits() -> None:
    frames = [make_frame(index, 120) for index in range(16)]
    for frame in frames[:8]:
        frame.usable = False
        frame.rejection_reason = TOO_BLURRY_REASON

    summary = summarize_usable_frames(
        frames,
        minimum_usable_frame_fraction=0.50,
        minimum_usable_frames=8,
    )

    assert summary.passed is True
    assert summary.frames_sampled == 16
    assert summary.frames_usable == 8
    assert summary.usable_fraction == 0.50
    assert summary.reason is None


def test_usable_frame_summary_fails_below_minimum_fraction() -> None:
    frames = [make_frame(index, 120) for index in range(10)]
    for frame in frames[:6]:
        frame.usable = False
        frame.rejection_reason = TOO_DARK_REASON

    summary = summarize_usable_frames(frames, 0.50, 4)

    assert summary.passed is False
    assert summary.usable_fraction == 0.40
    assert summary.reason == TOO_DARK_REASON
    assert summary.rejection_counts == {TOO_DARK_REASON: 6}


def test_usable_frame_summary_fails_below_absolute_count() -> None:
    frames = [make_frame(index, 120) for index in range(6)]

    summary = summarize_usable_frames(frames, 0.50, 8)

    assert summary.passed is False
    assert summary.frames_usable == 6
    assert summary.reason == NOT_ENOUGH_FRAMES_REASON


def test_usable_frame_summary_uses_dominant_rejection_reason() -> None:
    frames = [make_frame(index, 120) for index in range(10)]
    reasons = [TOO_BLURRY_REASON] * 4 + [TOO_DARK_REASON] * 2
    for frame, reason in zip(frames[:6], reasons, strict=True):
        frame.usable = False
        frame.rejection_reason = reason

    summary = summarize_usable_frames(frames, 0.50, 4)

    assert summary.passed is False
    assert summary.reason == TOO_BLURRY_REASON


def test_usable_frame_summary_handles_empty_list() -> None:
    summary = summarize_usable_frames([], 0.50, 8)

    assert summary.passed is False
    assert summary.usable_fraction == 0.0
    assert summary.reason == NOT_ENOUGH_FRAMES_REASON


def test_texture_gate_rejects_frame_with_too_few_features() -> None:
    frame = make_frame(0, 120)
    features = FeatureData(frame_index=0, keypoints=[object()] * 39)

    usable = apply_texture_gate(frame, features, minimum_feature_count=40)

    assert usable is False
    assert frame.rejection_reason == NOT_ENOUGH_TEXTURE_REASON


def test_texture_gate_accepts_exact_minimum_feature_count() -> None:
    frame = make_frame(0, 120)
    features = FeatureData(frame_index=0, keypoints=[object()] * 40)

    usable = apply_texture_gate(frame, features, minimum_feature_count=40)

    assert usable is True
    assert frame.rejection_reason is None


def test_texture_gate_rejects_mismatched_frame_data() -> None:
    frame = make_frame(0, 120)
    features = FeatureData(frame_index=1, keypoints=[object()] * 40)

    with pytest.raises(ValueError, match="does not belong"):
        apply_texture_gate(frame, features, minimum_feature_count=40)


def make_feature_matches(
    first_frame_index: int,
    second_frame_index: int,
    movement_px: float,
    count: int = 25,
) -> FeatureMatches:
    first_points = np.column_stack(
        (np.arange(count, dtype=np.float32), np.zeros(count, dtype=np.float32))
    )
    second_points = first_points + np.array([movement_px, 0], dtype=np.float32)
    return FeatureMatches(
        first_frame_index=first_frame_index,
        second_frame_index=second_frame_index,
        matches=[object()] * count,
        first_points=first_points,
        second_points=second_points,
    )


def test_calculate_median_feature_movement_in_pixels() -> None:
    matches = make_feature_matches(0, 1, movement_px=12.0)

    assert calculate_median_match_movement(matches) == pytest.approx(12.0)


def test_movement_gate_passes_at_exact_threshold() -> None:
    pairs = [
        make_feature_matches(0, 1, 6.0),
        make_feature_matches(1, 2, 8.0),
        make_feature_matches(2, 3, 10.0),
    ]

    summary = evaluate_camera_movement(pairs, 25, 8.0)

    assert summary.passed is True
    assert summary.median_movement_px == pytest.approx(8.0)
    assert summary.pair_movements_px == pytest.approx([6.0, 8.0, 10.0])
    assert summary.pairs_used == 3
    assert summary.matches_used == 75
    assert summary.reason is None


def test_movement_gate_rejects_too_little_movement() -> None:
    pairs = [
        make_feature_matches(0, 1, 3.0),
        make_feature_matches(1, 2, 5.0),
        make_feature_matches(2, 3, 7.0),
    ]

    summary = evaluate_camera_movement(pairs, 25, 8.0)

    assert summary.passed is False
    assert summary.median_movement_px == pytest.approx(5.0)
    assert summary.reason == NOT_ENOUGH_MOVEMENT_REASON


def test_movement_gate_ignores_pairs_with_too_few_matches() -> None:
    pairs = [make_feature_matches(0, 1, 20.0, count=24)]

    summary = evaluate_camera_movement(pairs, 25, 8.0)

    assert summary.passed is False
    assert summary.pairs_used == 0
    assert summary.reason == NOT_ENOUGH_MATCHES_REASON


def test_movement_gate_uses_pair_medians_so_dense_pair_does_not_dominate() -> None:
    pairs = [
        make_feature_matches(0, 1, 4.0, count=100),
        make_feature_matches(1, 2, 10.0, count=25),
        make_feature_matches(2, 3, 12.0, count=25),
    ]

    summary = evaluate_camera_movement(pairs, 25, 8.0)

    assert summary.median_movement_px == pytest.approx(10.0)
    assert summary.passed is True
