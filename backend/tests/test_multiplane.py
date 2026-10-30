"""Tests for second-plane, plane-normal, and box-corner analysis."""

from unittest.mock import patch

import numpy as np
import pytest

from boxcheck.geometry import (
    PlaneNormalComparison,
    analyze_second_plane,
    angle_between_plane_normals,
    build_camera_matrix,
    check_box_corner,
    compare_plane_normals,
    fit_homography,
)
from boxcheck.types import FeatureMatches


def make_matches(first: np.ndarray, second: np.ndarray) -> FeatureMatches:
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    return FeatureMatches(
        first_frame_index=0,
        second_frame_index=1,
        matches=[object()] * len(first),
        first_points=first,
        second_points=second,
    )


def transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    homogeneous = np.hstack((points, np.ones((len(points), 1))))
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:]


def make_two_plane_matches() -> FeatureMatches:
    random = np.random.default_rng(31)
    first_plane = random.uniform([80, 100], [300, 400], size=(70, 2))
    second_plane = random.uniform([305, 100], [540, 400], size=(45, 2))
    first_h = np.array([[1.0, 0.0, 14.0], [0.0, 1.0, 4.0], [0.0, 0.0, 1.0]])
    second_h = np.array(
        [[0.96, -0.10, 42.0], [0.10, 1.01, -24.0], [0.0002, 0.0, 1.0]]
    )
    first_target = transform(first_plane, first_h)
    second_target = transform(second_plane, second_h)
    first_target += random.normal(0, 0.1, first_target.shape)
    second_target += random.normal(0, 0.1, second_target.shape)
    return make_matches(
        np.vstack((first_plane, second_plane)),
        np.vstack((first_target, second_target)),
    )


def test_camera_matrix_uses_resolution_and_horizontal_fov() -> None:
    camera = build_camera_matrix(1280, 720, 64.0)

    expected_focal = 640 / np.tan(np.deg2rad(32.0))
    assert camera[0, 0] == pytest.approx(expected_focal)
    assert camera[1, 1] == pytest.approx(expected_focal)
    assert camera[0, 2] == 640
    assert camera[1, 2] == 360


def test_normal_angle_is_independent_of_normal_sign() -> None:
    assert angle_between_plane_normals([0, 0, 1], [0, 0, -1]) == pytest.approx(0.0)
    assert angle_between_plane_normals([0, 0, 1], [1, 0, 0]) == pytest.approx(90.0)


def test_normal_comparison_selects_compatible_camera_motion() -> None:
    identity = np.eye(3)
    first_solutions = [(identity, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))]
    second_solutions = [
        (identity, np.array([1.0, 0, 0]), np.array([1.0, 0, 0])),
    ]

    with patch(
        "boxcheck.geometry._decompose_homography",
        side_effect=[first_solutions, second_solutions],
    ):
        comparison = compare_plane_normals(identity, identity, identity, 0.15, 0.35)

    assert comparison.success is True
    assert comparison.angle_degrees == pytest.approx(90.0)


def test_box_corner_is_found_for_adjacent_plane_regions() -> None:
    first = np.array([[100, 100], [300, 100], [300, 400], [100, 400]])
    second = np.array([[305, 100], [500, 100], [500, 400], [305, 400]])

    result = check_box_corner(first, second, 640, 480, 0.80, 0.80, 20.0, 100.0)

    assert result.found is True
    assert result.edge_distance_px == pytest.approx(5.0)


def test_box_corner_rejects_separated_background_regions() -> None:
    first = np.array([[100, 100], [200, 100], [200, 300], [100, 300]])
    second = np.array([[350, 100], [500, 100], [500, 300], [350, 300]])

    result = check_box_corner(first, second, 640, 480, 0.80, 0.80, 20.0, 100.0)

    assert result.found is False
    assert result.edge_distance_px == pytest.approx(150.0)


def test_box_corner_rejects_contact_outside_central_roi() -> None:
    first = np.array([[0, 100], [20, 100], [20, 300], [0, 300]])
    second = np.array([[25, 100], [50, 100], [50, 300], [25, 300]])

    result = check_box_corner(first, second, 640, 480, 0.80, 0.80, 20.0, 100.0)

    assert result.found is False
    assert "outside" in result.reason


def test_second_plane_fit_maps_inliers_and_qualifies_box_surfaces() -> None:
    matches = make_two_plane_matches()
    first_model = fit_homography(matches, 3.0, 0.995, 5000, 20)
    normal_result = PlaneNormalComparison(
        True,
        45.0,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
        0.0,
        0.0,
        None,
    )

    with patch("boxcheck.geometry.compare_plane_normals", return_value=normal_result):
        result = analyze_second_plane(
            matches,
            first_model,
            build_camera_matrix(640, 480, 64.0),
            640,
            480,
            0.80,
            0.80,
            3.0,
            0.995,
            5000,
            12,
            25.0,
            0.15,
            0.35,
            20.0,
            100.0,
        )

    assert result.model_found is True
    assert result.model.inlier_count >= 12
    assert int(result.global_inlier_mask.sum()) == result.model.inlier_count
    assert result.orientation_distinct is True
    assert result.corner.found is True
    assert result.qualifies_as_box_surfaces is True


def test_second_plane_fails_when_too_few_points_remain() -> None:
    random = np.random.default_rng(4)
    first = random.uniform([50, 50], [550, 430], size=(40, 2))
    second = first + np.array([12.0, 3.0])
    matches = make_matches(first, second)
    first_model = fit_homography(matches, 3.0, 0.995, 5000, 20)

    result = analyze_second_plane(
        matches,
        first_model,
        build_camera_matrix(640, 480, 64.0),
        640,
        480,
        0.80,
        0.80,
        3.0,
        0.995,
        5000,
        12,
        25.0,
        0.15,
        0.35,
        20.0,
        100.0,
    )

    assert result.model_found is False
    assert result.qualifies_as_box_surfaces is False
    assert "Too few" in result.reason


def test_similar_plane_normals_do_not_qualify() -> None:
    matches = make_two_plane_matches()
    first_model = fit_homography(matches, 3.0, 0.995, 5000, 20)
    normal_result = PlaneNormalComparison(
        True,
        10.0,
        np.array([0.0, 0.0, 1.0]),
        np.array([0.17, 0.0, 0.98]),
        0.0,
        0.0,
        None,
    )

    with patch("boxcheck.geometry.compare_plane_normals", return_value=normal_result):
        result = analyze_second_plane(
            matches,
            first_model,
            build_camera_matrix(640, 480, 64.0),
            640,
            480,
            0.80,
            0.80,
            3.0,
            0.995,
            5000,
            12,
            25.0,
            0.15,
            0.35,
            20.0,
            100.0,
        )

    assert result.model_found is True
    assert result.orientation_distinct is False
    assert result.qualifies_as_box_surfaces is False
    assert result.reason == "Plane normals are too similar"
