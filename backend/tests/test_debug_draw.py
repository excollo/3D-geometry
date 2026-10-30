"""Tests for annotated geometry debug images."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from boxcheck.debug_draw import (
    HEADER_HEIGHT_PX,
    OUTLIER_COLOR,
    PLANE_ONE_COLOR,
    PLANE_TWO_COLOR,
    DebugDrawError,
    DebugPairInput,
    annotate_pair,
    save_debug_pair_images,
)
from boxcheck.types import FeatureMatches, FrameData, PairMetrics


def make_debug_input() -> DebugPairInput:
    first_image = np.full((60, 80), 80, dtype=np.uint8)
    second_image = np.full((60, 80), 100, dtype=np.uint8)
    first = FrameData(
        0,
        0.0,
        cv2.cvtColor(first_image, cv2.COLOR_GRAY2BGR),
        first_image,
    )
    second = FrameData(
        1,
        1.0,
        cv2.cvtColor(second_image, cv2.COLOR_GRAY2BGR),
        second_image,
    )
    matches = FeatureMatches(
        0,
        1,
        [object(), object(), object()],
        np.array([[10, 15], [30, 25], [50, 35]], dtype=np.float32),
        np.array([[14, 15], [34, 25], [54, 35]], dtype=np.float32),
    )
    metrics = PairMetrics(
        0,
        1,
        1.0,
        match_count=3,
        median_movement_px=4.0,
        homography_inlier_ratio=0.5,
        fundamental_inlier_ratio=0.8,
        median_parallax_px=3.0,
        pair_score=0.7,
    )
    return DebugPairInput(
        first,
        second,
        matches,
        np.array([True, False, False]),
        np.array([False, True, False]),
        metrics,
    )


def test_annotated_pair_has_header_and_side_by_side_frames() -> None:
    image = annotate_pair(make_debug_input())

    assert image.shape == (HEADER_HEIGHT_PX + 60, 160, 3)
    assert image.dtype == np.uint8


def test_annotated_pair_uses_distinct_plane_and_outlier_colors() -> None:
    debug_input = make_debug_input()
    image = annotate_pair(debug_input)
    y_values = [15, 25, 35]
    x_values = [10, 30, 50]
    expected = [PLANE_ONE_COLOR, PLANE_TWO_COLOR, OUTLIER_COLOR]

    for x, y, color in zip(x_values, y_values, expected, strict=True):
        assert tuple(image[y + HEADER_HEIGHT_PX, x]) == color


def test_invalid_plane_mask_length_is_rejected() -> None:
    debug_input = make_debug_input()
    debug_input.first_plane_mask = np.array([True])

    with pytest.raises(DebugDrawError, match="one value per feature match"):
        annotate_pair(debug_input)


def test_overlapping_plane_masks_are_rejected() -> None:
    debug_input = make_debug_input()
    debug_input.second_plane_mask[0] = True

    with pytest.raises(DebugDrawError, match="both plane"):
        annotate_pair(debug_input)


def test_frame_index_mismatch_is_rejected() -> None:
    debug_input = make_debug_input()
    debug_input.second_frame.index = 2

    with pytest.raises(DebugDrawError, match="Second frame"):
        annotate_pair(debug_input)


def test_save_debug_images_respects_maximum_and_returns_paths(tmp_path: Path) -> None:
    debug_pairs = [make_debug_input(), make_debug_input(), make_debug_input()]

    paths = save_debug_pair_images(tmp_path / "debug", debug_pairs, maximum_images=2)

    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
    assert all(cv2.imread(str(path)) is not None for path in paths)


def test_zero_maximum_saves_nothing(tmp_path: Path) -> None:
    output = tmp_path / "debug"

    paths = save_debug_pair_images(output, [make_debug_input()], maximum_images=0)

    assert paths == []
    assert not output.exists()
