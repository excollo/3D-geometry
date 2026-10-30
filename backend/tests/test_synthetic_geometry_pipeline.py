"""End-to-end synthetic verdict tests for the standalone geometry checker."""

import numpy as np

from boxcheck.geometry import (
    analyze_pair_geometry,
    analyze_second_plane,
    build_camera_matrix,
    fit_homography,
)
from boxcheck.scoring import aggregate_pair_measurements, score_geometry_pair
from boxcheck.types import FeatureMatches, FramePair, Verdict


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
HORIZONTAL_FOV = 64.0
CAMERA = build_camera_matrix(IMAGE_WIDTH, IMAGE_HEIGHT, HORIZONTAL_FOV)


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


def project_points(
    points_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    camera_points = (rotation @ points_3d.T).T + translation
    homogeneous_pixels = (CAMERA @ camera_points.T).T
    return homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2:]


def second_camera_pose() -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(4.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    return rotation, np.array([1.0, 0.02, 0.03])


def make_flat_plane_matches(noise_px: float = 0.15) -> FeatureMatches:
    random = np.random.default_rng(101)
    x_values = np.linspace(-2.0, 2.0, 12)
    y_values = np.linspace(-1.4, 1.4, 9)
    points = np.array([(x, y, 6.0) for y in y_values for x in x_values])
    rotation, translation = second_camera_pose()
    first = project_points(points, np.eye(3), np.zeros(3))
    second = project_points(points, rotation, translation)
    first += random.normal(0, noise_px, first.shape)
    second += random.normal(0, noise_px, second.shape)
    return make_matches(first, second)


def make_two_face_box_matches(noise_px: float = 0.12) -> FeatureMatches:
    random = np.random.default_rng(202)
    y_values = np.linspace(-1.4, 1.4, 9)
    front_x_values = np.linspace(-2.0, 0.0, 11)
    front = np.array(
        [(x, y, 6.0) for y in y_values for x in front_x_values]
    )

    # The second face starts beside the front face's x=0 edge and recedes in depth.
    side_distances = np.linspace(0.06, 2.0, 8)
    side = np.array(
        [
            (distance / np.sqrt(2), y, 6.0 + distance / np.sqrt(2))
            for y in y_values
            for distance in side_distances
        ]
    )
    points = np.vstack((front, side))
    rotation, translation = second_camera_pose()
    first = project_points(points, np.eye(3), np.zeros(3))
    second = project_points(points, rotation, translation)
    first += random.normal(0, noise_px, first.shape)
    second += random.normal(0, noise_px, second.shape)
    return make_matches(first, second)


def analyze_synthetic_pair(matches: FeatureMatches):
    geometry = analyze_pair_geometry(
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
    second_plane = analyze_second_plane(
        matches,
        geometry.homography,
        CAMERA,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
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
    pair_metrics = score_geometry_pair(
        FramePair(0, 1, 1.0),
        geometry,
        second_plane,
        0.40,
        0.35,
        0.25,
        2.5,
        12,
        0.15,
    )
    # Three consistent observations provide full confidence for this synthetic video.
    decision = aggregate_pair_measurements(
        [pair_metrics, pair_metrics, pair_metrics],
        frames_sampled=12,
        frames_used=12,
        median_brightness=128.0,
        median_blur=100.0,
        three_d_score_threshold=0.65,
        flat_score_threshold=0.35,
        minimum_confidence=0.60,
        minimum_median_movement_px=8.0,
        minimum_pairs_for_full_confidence=3,
        confidence_mad_scale=2.0,
    )
    return geometry, second_plane, decision


def test_projected_flat_plane_produces_flat_verdict() -> None:
    geometry, second_plane, decision = analyze_synthetic_pair(
        make_flat_plane_matches()
    )

    assert geometry.homography.inlier_ratio > 0.95
    assert geometry.homography_gric < geometry.fundamental_gric
    assert second_plane.model_found is False
    assert decision.verdict is Verdict.FLAT


def test_two_physical_box_faces_produce_3d_verdict() -> None:
    geometry, second_plane, decision = analyze_synthetic_pair(
        make_two_face_box_matches()
    )

    assert geometry.fundamental.success is True
    assert second_plane.model_found is True
    assert second_plane.orientation_distinct is True
    assert second_plane.corner.found is True
    assert second_plane.qualifies_as_box_surfaces is True
    assert decision.verdict is Verdict.THREE_D


def test_too_few_synthetic_points_produce_insufficient_verdict() -> None:
    matches = make_matches(
        np.array([[100, 100], [200, 100], [100, 200]], dtype=np.float32),
        np.array([[110, 100], [210, 100], [110, 200]], dtype=np.float32),
    )
    geometry = analyze_pair_geometry(
        matches,
        3.0,
        0.995,
        5000,
        20,
        20,
        2.5,
        8.0,
    )
    decision = aggregate_pair_measurements(
        [], 3, 3, 128.0, 100.0, 0.65, 0.35, 0.60, 8.0, 3, 2.0
    )

    assert geometry.homography.success is False
    assert geometry.fundamental.success is False
    assert decision.verdict is Verdict.INSUFFICIENT


def test_flat_homography_remains_stable_with_random_outliers() -> None:
    original = make_flat_plane_matches()
    random = np.random.default_rng(303)
    outlier_count = 10
    first = np.vstack(
        (original.first_points, random.uniform([40, 40], [600, 440], (outlier_count, 2)))
    )
    second = np.vstack(
        (original.second_points, random.uniform([40, 40], [600, 440], (outlier_count, 2)))
    )
    contaminated = make_matches(first, second)

    fit = fit_homography(contaminated, 3.0, 0.995, 5000, 20)

    assert fit.success is True
    assert fit.inlier_count >= len(original.matches) - 2


def test_synthetic_generation_is_repeatable() -> None:
    first_run = make_two_face_box_matches()
    second_run = make_two_face_box_matches()

    np.testing.assert_allclose(first_run.first_points, second_run.first_points)
    np.testing.assert_allclose(first_run.second_points, second_run.second_points)
