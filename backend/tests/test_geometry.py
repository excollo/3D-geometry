"""Synthetic tests for planar-versus-3D pair geometry."""

import numpy as np
import pytest

from boxcheck.geometry import (
    analyze_pair_geometry,
    calculate_gric,
    fit_fundamental_matrix,
    fit_homography,
    gric_3d_preference,
    sampson_errors,
    symmetric_homography_errors,
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


def make_planar_matches(count: int = 100, noise: float = 0.15) -> FeatureMatches:
    random = np.random.default_rng(11)
    first = random.uniform([40, 40], [600, 440], size=(count, 2))
    homography = np.array(
        [[1.01, 0.015, 18.0], [-0.01, 0.99, 9.0], [0.00005, -0.00004, 1.0]]
    )
    homogeneous = np.hstack((first, np.ones((count, 1))))
    projected = (homography @ homogeneous.T).T
    second = projected[:, :2] / projected[:, 2:]
    second += random.normal(0, noise, second.shape)
    return make_matches(first, second)


def make_3d_matches(count: int = 160, noise: float = 0.15) -> FeatureMatches:
    random = np.random.default_rng(21)
    points = np.column_stack(
        (
            random.uniform(-1.8, 1.8, count),
            random.uniform(-1.2, 1.2, count),
            random.uniform(4.0, 9.0, count),
        )
    )
    focal = 700.0
    camera = np.array([[focal, 0, 320], [0, focal, 240], [0, 0, 1.0]])
    angle = np.deg2rad(5.0)
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    translation = np.array([0.45, 0.03, 0.0])

    def project(world, rotation_matrix, translation_vector):
        camera_points = (rotation_matrix @ world.T).T + translation_vector
        pixels = (camera @ camera_points.T).T
        return pixels[:, :2] / pixels[:, 2:]

    first = project(points, np.eye(3), np.zeros(3))
    second = project(points, rotation, translation)
    first += random.normal(0, noise, first.shape)
    second += random.normal(0, noise, second.shape)
    return make_matches(first, second)


def analyze(matches: FeatureMatches):
    return analyze_pair_geometry(
        matches,
        ransac_threshold_px=3.0,
        confidence=0.995,
        max_iterations=5000,
        minimum_homography_inliers=20,
        minimum_fundamental_inliers=20,
        minimum_parallax_px=2.5,
        minimum_median_movement_px=8.0,
        noise_sigma_px=1.0,
    )


def test_symmetric_homography_error_is_near_zero_for_exact_transform() -> None:
    matches = make_planar_matches(noise=0.0)
    fit = fit_homography(matches, 3.0, 0.995, 5000, 20)

    errors = symmetric_homography_errors(
        matches.first_points, matches.second_points, fit.matrix
    )

    assert float(np.median(errors)) < 1e-6


def test_sampson_error_is_finite_for_valid_fundamental_fit() -> None:
    matches = make_3d_matches()
    fit = fit_fundamental_matrix(matches, 3.0, 0.995, 5000, 20)

    errors = sampson_errors(matches.first_points, matches.second_points, fit.matrix)

    assert np.isfinite(errors).all()
    assert float(np.median(errors)) < 1.0


def test_gric_prefers_smaller_residuals_with_same_model_complexity() -> None:
    small = calculate_gric(np.full(50, 0.2), 1.0, 2, 8)
    large = calculate_gric(np.full(50, 4.0), 1.0, 2, 8)

    assert small < large


def test_gric_preference_scale_has_expected_direction() -> None:
    assert gric_3d_preference(1000, 500) > 0.5
    assert gric_3d_preference(500, 1000) < 0.5
    assert gric_3d_preference(500, 500) == 0.5


def test_planar_points_favor_homography_and_have_low_parallax() -> None:
    result = analyze(make_planar_matches())

    assert result.homography.success is True
    assert result.homography.inlier_ratio > 0.95
    assert result.homography_gric < result.fundamental_gric
    assert result.gric_3d_preference < 0.5
    assert result.parallax.median_px < 2.5


def test_multi_depth_points_favor_fundamental_model_and_leave_parallax() -> None:
    result = analyze(make_3d_matches())

    assert result.fundamental.success is True
    assert result.fundamental.inlier_ratio > result.homography.inlier_ratio
    assert result.fundamental_gric < result.homography_gric
    assert result.gric_3d_preference > 0.5
    assert result.parallax.median_px > 2.5
    assert result.parallax.point_count >= 20


def test_too_few_points_return_controlled_model_failures() -> None:
    matches = make_matches(
        np.array([[0, 0], [1, 0], [0, 1]]),
        np.array([[2, 0], [3, 0], [2, 1]]),
    )

    result = analyze(matches)

    assert result.homography.success is False
    assert "at least 4" in result.homography.reason
    assert result.fundamental.success is False
    assert "at least 8" in result.fundamental.reason
    assert result.homography_gric is None
    assert result.fundamental_gric is None


def test_non_finite_points_are_rejected() -> None:
    first = np.zeros((8, 2), dtype=np.float32)
    second = np.zeros((8, 2), dtype=np.float32)
    second[0, 0] = np.nan
    matches = make_matches(first, second)

    with pytest.raises(ValueError, match="finite"):
        analyze(matches)


def test_repeated_degenerate_points_fail_without_crashing() -> None:
    first = np.full((25, 2), [100, 100], dtype=np.float32)
    second = np.full((25, 2), [110, 100], dtype=np.float32)

    result = analyze(make_matches(first, second))

    assert result.homography.success is False
    assert result.fundamental.success is False


def test_singular_homography_error_is_controlled() -> None:
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)

    with pytest.raises(ValueError, match="singular"):
        symmetric_homography_errors(points, points, np.zeros((3, 3)))
