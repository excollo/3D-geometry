"""Compare planar and general-3D geometry for matched frame points.

GRIC follows the Torr-style robust information criterion with measurement
dimension r=4, manifold dimension d=2 and k=8 parameters for a homography,
and d=3 and k=7 for a fundamental matrix. Residual inputs are squared pixel
errors and ``noise_sigma_px`` supplies their expected scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from boxcheck.types import FeatureMatches


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class GeometryModelResult:
    """Fit, residuals, and RANSAC support for one geometric model."""

    success: bool
    matrix: FloatArray | None
    inlier_mask: BoolArray
    residuals_squared_px: FloatArray
    inlier_count: int
    inlier_ratio: float
    reason: str | None = None


@dataclass(slots=True)
class ParallaxResult:
    """Homography residuals supported by the fundamental model."""

    median_px: float
    percentile_75_px: float
    point_count: int


@dataclass(slots=True)
class PairGeometryResult:
    """Complete planar-versus-3D analysis for one matched frame pair."""

    first_frame_index: int
    second_frame_index: int
    homography: GeometryModelResult
    fundamental: GeometryModelResult
    homography_gric: float | None
    fundamental_gric: float | None
    gric_3d_preference: float
    parallax: ParallaxResult
    median_movement_px: float
    low_motion: bool
    low_parallax: bool
    pure_rotation_risk: bool


@dataclass(slots=True)
class PlaneNormalComparison:
    """Most camera-consistent normal pair from two homography decompositions."""

    success: bool
    angle_degrees: float | None
    first_normal: FloatArray | None
    second_normal: FloatArray | None
    rotation_difference: float | None
    translation_direction_difference: float | None
    reason: str | None = None


@dataclass(slots=True)
class BoxCornerResult:
    """Spatial relationship between two plane point regions in frame one."""

    found: bool
    edge_distance_px: float | None
    contact_point: FloatArray | None
    first_hull_area_px2: float
    second_hull_area_px2: float
    reason: str | None = None


@dataclass(slots=True)
class SecondPlaneResult:
    """Second-plane model plus orientation and box-corner evidence."""

    model_found: bool
    model: GeometryModelResult
    global_inlier_mask: BoolArray
    normal_comparison: PlaneNormalComparison
    orientation_distinct: bool
    corner: BoxCornerResult
    qualifies_as_box_surfaces: bool
    reason: str


def validate_matched_points(matches: FeatureMatches) -> None:
    """Reject malformed, non-finite, or duplicated point-array input."""
    first = np.asarray(matches.first_points)
    second = np.asarray(matches.second_points)
    expected_shape = (len(matches.matches), 2)
    if first.shape != expected_shape or second.shape != expected_shape:
        raise ValueError(f"Matched point arrays must both have shape {expected_shape}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Matched point coordinates must be finite")


def symmetric_homography_errors(
    first_points: NDArray[np.floating],
    second_points: NDArray[np.floating],
    homography: NDArray[np.floating],
) -> FloatArray:
    """Return mean forward/backward squared transfer error per match."""
    first, second = _validated_point_arrays(first_points, second_points)
    matrix = _validated_matrix(homography, "homography")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("Homography matrix is singular") from error

    forward = _perspective_transform(first, matrix)
    backward = _perspective_transform(second, inverse)
    forward_squared = np.sum((forward - second) ** 2, axis=1)
    backward_squared = np.sum((backward - first) ** 2, axis=1)
    return (forward_squared + backward_squared) / 2.0


def sampson_errors(
    first_points: NDArray[np.floating],
    second_points: NDArray[np.floating],
    fundamental_matrix: NDArray[np.floating],
) -> FloatArray:
    """Return first-order squared epipolar distance for every match."""
    first, second = _validated_point_arrays(first_points, second_points)
    matrix = _validated_matrix(fundamental_matrix, "fundamental matrix")
    ones = np.ones((len(first), 1), dtype=np.float64)
    first_h = np.hstack((first, ones))
    second_h = np.hstack((second, ones))
    f_first = (matrix @ first_h.T).T
    ft_second = (matrix.T @ second_h.T).T
    numerator = np.sum(second_h * f_first, axis=1) ** 2
    denominator = (
        f_first[:, 0] ** 2
        + f_first[:, 1] ** 2
        + ft_second[:, 0] ** 2
        + ft_second[:, 1] ** 2
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.inf),
        where=denominator > np.finfo(np.float64).eps,
    )


def fit_homography(
    matches: FeatureMatches,
    ransac_threshold_px: float,
    confidence: float,
    max_iterations: int,
    minimum_inliers: int,
) -> GeometryModelResult:
    """Fit a one-plane homography with RANSAC."""
    validate_matched_points(matches)
    _validate_ransac_settings(
        ransac_threshold_px, confidence, max_iterations, minimum_inliers
    )
    point_count = len(matches.matches)
    if point_count < 4:
        return _failed_model(point_count, "Homography requires at least 4 matches")

    try:
        matrix, mask = cv2.findHomography(
            matches.first_points,
            matches.second_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
            maxIters=max_iterations,
            confidence=confidence,
        )
    except cv2.error as error:
        return _failed_model(point_count, f"OpenCV homography failure: {error}")

    if matrix is None or mask is None or np.asarray(matrix).shape != (3, 3):
        return _failed_model(point_count, "OpenCV could not estimate a homography")

    try:
        residuals = symmetric_homography_errors(
            matches.first_points, matches.second_points, matrix
        )
    except ValueError as error:
        return _failed_model(point_count, str(error))

    return _completed_model(matrix, mask, residuals, minimum_inliers, "homography")


def fit_fundamental_matrix(
    matches: FeatureMatches,
    ransac_threshold_px: float,
    confidence: float,
    max_iterations: int,
    minimum_inliers: int,
) -> GeometryModelResult:
    """Fit a general two-view fundamental matrix with RANSAC."""
    validate_matched_points(matches)
    _validate_ransac_settings(
        ransac_threshold_px, confidence, max_iterations, minimum_inliers
    )
    point_count = len(matches.matches)
    if point_count < 8:
        return _failed_model(
            point_count, "Fundamental matrix requires at least 8 matches"
        )

    try:
        matrix, mask = cv2.findFundamentalMat(
            matches.first_points,
            matches.second_points,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=ransac_threshold_px,
            confidence=confidence,
            maxIters=max_iterations,
        )
    except cv2.error as error:
        return _failed_model(point_count, f"OpenCV fundamental-matrix failure: {error}")

    if matrix is None or mask is None or np.asarray(matrix).shape != (3, 3):
        return _failed_model(
            point_count, "OpenCV could not estimate a fundamental matrix"
        )

    residuals = sampson_errors(matches.first_points, matches.second_points, matrix)
    return _completed_model(
        matrix, mask, residuals, minimum_inliers, "fundamental matrix"
    )


def calculate_gric(
    residuals_squared_px: NDArray[np.floating],
    noise_sigma_px: float,
    manifold_dimension: int,
    parameter_count: int,
    measurement_dimension: int = 4,
) -> float:
    """Calculate robust geometric information criterion; lower is better."""
    residuals = np.asarray(residuals_squared_px, dtype=np.float64)
    if residuals.ndim != 1 or len(residuals) == 0:
        raise ValueError("GRIC requires a non-empty one-dimensional residual array")
    if noise_sigma_px <= 0:
        raise ValueError("noise_sigma_px must be greater than zero")
    if not 0 < manifold_dimension < measurement_dimension:
        raise ValueError("manifold_dimension must be between 0 and measurement_dimension")
    if parameter_count < 1:
        raise ValueError("parameter_count must be at least 1")

    finite_residuals = np.nan_to_num(
        residuals,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    )
    lambda_1 = np.log(measurement_dimension)
    lambda_2 = np.log(measurement_dimension * len(finite_residuals))
    lambda_3 = 2.0
    robust_cap = lambda_3 * (measurement_dimension - manifold_dimension)
    data_cost = np.minimum(
        finite_residuals / (noise_sigma_px**2),
        robust_cap,
    ).sum()
    complexity_cost = (
        lambda_1 * manifold_dimension * len(finite_residuals)
        + lambda_2 * parameter_count
    )
    return float(data_cost + complexity_cost)


def gric_3d_preference(homography_gric: float, fundamental_gric: float) -> float:
    """Map relative GRIC difference to 0=planar, 0.5=tie, 1=general 3D."""
    if not np.isfinite(homography_gric) or not np.isfinite(fundamental_gric):
        raise ValueError("GRIC values must be finite")
    scale = abs(homography_gric) + abs(fundamental_gric)
    if scale <= np.finfo(np.float64).eps:
        return 0.5
    score = 0.5 + (homography_gric - fundamental_gric) / (2.0 * scale)
    return float(np.clip(score, 0.0, 1.0))


def calculate_residual_parallax(
    homography: GeometryModelResult,
    fundamental: GeometryModelResult,
    homography_outlier_threshold_px: float,
) -> ParallaxResult:
    """Measure H residual only where F agrees and H does not."""
    if homography_outlier_threshold_px <= 0:
        raise ValueError("homography_outlier_threshold_px must be greater than zero")
    if homography.matrix is None or fundamental.matrix is None:
        return ParallaxResult(0.0, 0.0, 0)

    homography_errors_px = np.sqrt(homography.residuals_squared_px)
    candidate_mask = (
        fundamental.inlier_mask
        & np.isfinite(homography_errors_px)
        & (homography_errors_px > homography_outlier_threshold_px)
    )
    values = homography_errors_px[candidate_mask]
    if len(values) == 0:
        return ParallaxResult(0.0, 0.0, 0)
    return ParallaxResult(
        median_px=float(np.median(values)),
        percentile_75_px=float(np.percentile(values, 75)),
        point_count=len(values),
    )


def analyze_pair_geometry(
    matches: FeatureMatches,
    ransac_threshold_px: float,
    confidence: float,
    max_iterations: int,
    minimum_homography_inliers: int,
    minimum_fundamental_inliers: int,
    minimum_parallax_px: float,
    minimum_median_movement_px: float,
    noise_sigma_px: float | None = None,
) -> PairGeometryResult:
    """Run H, F, GRIC, parallax, and safety measurements for one pair."""
    validate_matched_points(matches)
    if minimum_parallax_px < 0 or minimum_median_movement_px < 0:
        raise ValueError("Movement and parallax thresholds must be non-negative")

    homography = fit_homography(
        matches,
        ransac_threshold_px,
        confidence,
        max_iterations,
        minimum_homography_inliers,
    )
    fundamental = fit_fundamental_matrix(
        matches,
        ransac_threshold_px,
        confidence,
        max_iterations,
        minimum_fundamental_inliers,
    )

    sigma = noise_sigma_px if noise_sigma_px is not None else ransac_threshold_px
    homography_gric = None
    fundamental_gric = None
    preference = 0.5
    if homography.matrix is not None:
        homography_gric = calculate_gric(
            homography.residuals_squared_px, sigma, 2, 8
        )
    if fundamental.matrix is not None:
        fundamental_gric = calculate_gric(
            fundamental.residuals_squared_px, sigma, 3, 7
        )
    if homography_gric is not None and fundamental_gric is not None:
        preference = gric_3d_preference(homography_gric, fundamental_gric)

    parallax = calculate_residual_parallax(
        homography,
        fundamental,
        ransac_threshold_px,
    )
    movement = _median_movement(matches)
    low_motion = movement < minimum_median_movement_px
    low_parallax = parallax.median_px < minimum_parallax_px
    homography_wins = (
        homography_gric is not None
        and fundamental_gric is not None
        and homography_gric < fundamental_gric
    )

    return PairGeometryResult(
        first_frame_index=matches.first_frame_index,
        second_frame_index=matches.second_frame_index,
        homography=homography,
        fundamental=fundamental,
        homography_gric=homography_gric,
        fundamental_gric=fundamental_gric,
        gric_3d_preference=preference,
        parallax=parallax,
        median_movement_px=movement,
        low_motion=low_motion,
        low_parallax=low_parallax,
        pure_rotation_risk=homography_wins and low_motion,
    )


def build_camera_matrix(
    image_width_px: int,
    image_height_px: int,
    horizontal_field_of_view_degrees: float,
) -> FloatArray:
    """Build an approximate pinhole camera matrix from image size and HFOV."""
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("Image width and height must be greater than zero")
    if not 1.0 < horizontal_field_of_view_degrees < 179.0:
        raise ValueError("Horizontal field of view must be between 1 and 179 degrees")

    half_angle_radians = np.deg2rad(horizontal_field_of_view_degrees / 2.0)
    focal_px = (image_width_px / 2.0) / np.tan(half_angle_radians)
    return np.array(
        [
            [focal_px, 0.0, image_width_px / 2.0],
            [0.0, focal_px, image_height_px / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def compare_plane_normals(
    first_homography: NDArray[np.floating],
    second_homography: NDArray[np.floating],
    camera_matrix: NDArray[np.floating],
    maximum_rotation_difference: float,
    maximum_translation_direction_difference: float,
) -> PlaneNormalComparison:
    """Compare normals from decomposition solutions with compatible camera motion."""
    if maximum_rotation_difference < 0:
        raise ValueError("maximum_rotation_difference must be non-negative")
    if not 0 <= maximum_translation_direction_difference <= 1:
        raise ValueError(
            "maximum_translation_direction_difference must be between 0 and 1"
        )

    first_matrix = _validated_matrix(first_homography, "first homography")
    second_matrix = _validated_matrix(second_homography, "second homography")
    calibration = _validated_matrix(camera_matrix, "camera matrix")
    first_solutions = _decompose_homography(first_matrix, calibration)
    second_solutions = _decompose_homography(second_matrix, calibration)
    if not first_solutions or not second_solutions:
        return PlaneNormalComparison(
            False, None, None, None, None, None, "Homography decomposition failed"
        )

    candidates = []
    for first_rotation, first_translation, first_normal in first_solutions:
        for second_rotation, second_translation, second_normal in second_solutions:
            rotation_difference = float(
                np.linalg.norm(first_rotation - second_rotation, ord="fro")
            )
            translation_difference = _direction_difference(
                first_translation, second_translation
            )
            candidates.append(
                (
                    rotation_difference + translation_difference,
                    rotation_difference,
                    translation_difference,
                    first_normal,
                    second_normal,
                )
            )

    _, rotation_difference, translation_difference, first_normal, second_normal = min(
        candidates, key=lambda candidate: candidate[0]
    )
    if (
        rotation_difference > maximum_rotation_difference
        or translation_difference > maximum_translation_direction_difference
    ):
        return PlaneNormalComparison(
            False,
            None,
            None,
            None,
            rotation_difference,
            translation_difference,
            "No compatible camera-motion solution for both planes",
        )

    angle = angle_between_plane_normals(first_normal, second_normal)
    return PlaneNormalComparison(
        True,
        angle,
        first_normal,
        second_normal,
        rotation_difference,
        translation_difference,
        None,
    )


def angle_between_plane_normals(
    first_normal: NDArray[np.floating],
    second_normal: NDArray[np.floating],
) -> float:
    """Return the sign-independent acute angle between two plane normals."""
    first = _normalized_vector(first_normal, "first normal")
    second = _normalized_vector(second_normal, "second normal")
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def check_box_corner(
    first_plane_points: NDArray[np.floating],
    second_plane_points: NDArray[np.floating],
    image_width_px: int,
    image_height_px: int,
    roi_width_fraction: float,
    roi_height_fraction: float,
    edge_tolerance_px: float,
    minimum_hull_area_px2: float,
) -> BoxCornerResult:
    """Require two meaningful plane regions to touch inside the central ROI."""
    first, _ = _validated_point_arrays(first_plane_points, first_plane_points)
    second, _ = _validated_point_arrays(second_plane_points, second_plane_points)
    if len(first) < 3 or len(second) < 3:
        return BoxCornerResult(
            False, None, None, 0.0, 0.0, "Each plane needs at least 3 points"
        )
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("Image width and height must be greater than zero")
    if not 0 < roi_width_fraction <= 1 or not 0 < roi_height_fraction <= 1:
        raise ValueError("ROI fractions must be greater than 0 and at most 1")
    if edge_tolerance_px < 0 or minimum_hull_area_px2 < 0:
        raise ValueError("Corner distance and hull area thresholds must be non-negative")

    first_hull = cv2.convexHull(first.astype(np.float32)).reshape(-1, 2)
    second_hull = cv2.convexHull(second.astype(np.float32)).reshape(-1, 2)
    first_area = float(cv2.contourArea(first_hull))
    second_area = float(cv2.contourArea(second_hull))
    if first_area < minimum_hull_area_px2 or second_area < minimum_hull_area_px2:
        return BoxCornerResult(
            False,
            None,
            None,
            first_area,
            second_area,
            "A plane region is too small to establish a box face",
        )

    distance, first_contact, second_contact = _closest_hull_points(
        first_hull, second_hull
    )
    contact = (first_contact + second_contact) / 2.0
    inside_roi = _point_inside_central_roi(
        contact,
        image_width_px,
        image_height_px,
        roi_width_fraction,
        roi_height_fraction,
    )
    found = bool(distance <= edge_tolerance_px and inside_roi)
    if distance > edge_tolerance_px:
        reason = "Plane regions do not meet within the edge tolerance"
    elif not inside_roi:
        reason = "Plane contact lies outside the central ROI"
    else:
        reason = None

    return BoxCornerResult(
        found,
        distance,
        contact,
        first_area,
        second_area,
        reason,
    )


def analyze_second_plane(
    matches: FeatureMatches,
    first_homography: GeometryModelResult,
    camera_matrix: NDArray[np.floating],
    image_width_px: int,
    image_height_px: int,
    roi_width_fraction: float,
    roi_height_fraction: float,
    ransac_threshold_px: float,
    confidence: float,
    max_iterations: int,
    minimum_second_plane_inliers: int,
    minimum_normal_angle_degrees: float,
    maximum_rotation_difference: float,
    maximum_translation_direction_difference: float,
    edge_tolerance_px: float,
    minimum_hull_area_px2: float,
) -> SecondPlaneResult:
    """Fit and validate a second plane as credible adjacent box-face evidence."""
    validate_matched_points(matches)
    point_count = len(matches.matches)
    empty_global_mask = np.zeros(point_count, dtype=bool)
    unavailable_normal = PlaneNormalComparison(
        False, None, None, None, None, None, "Second plane unavailable"
    )
    unavailable_corner = BoxCornerResult(
        False, None, None, 0.0, 0.0, "Second plane unavailable"
    )
    if first_homography.matrix is None or len(first_homography.inlier_mask) != point_count:
        return SecondPlaneResult(
            False,
            _failed_model(0, "First homography is unavailable"),
            empty_global_mask,
            unavailable_normal,
            False,
            unavailable_corner,
            False,
            "First plane could not be established",
        )

    remaining_indices = np.flatnonzero(~first_homography.inlier_mask)
    if len(remaining_indices) < 4:
        return SecondPlaneResult(
            False,
            _failed_model(len(remaining_indices), "Too few points remain for plane 2"),
            empty_global_mask,
            unavailable_normal,
            False,
            unavailable_corner,
            False,
            "Too few points remain after removing plane 1",
        )

    remaining_matches = _subset_matches(matches, remaining_indices)
    second_model = fit_homography(
        remaining_matches,
        ransac_threshold_px,
        confidence,
        max_iterations,
        minimum_second_plane_inliers,
    )
    global_mask = np.zeros(point_count, dtype=bool)
    if len(second_model.inlier_mask) == len(remaining_indices):
        global_mask[remaining_indices[second_model.inlier_mask]] = True

    if not second_model.success or second_model.matrix is None:
        return SecondPlaneResult(
            False,
            second_model,
            global_mask,
            unavailable_normal,
            False,
            unavailable_corner,
            False,
            second_model.reason or "Second homography could not be established",
        )

    normal_comparison = compare_plane_normals(
        first_homography.matrix,
        second_model.matrix,
        camera_matrix,
        maximum_rotation_difference,
        maximum_translation_direction_difference,
    )
    orientation_distinct = (
        normal_comparison.success
        and normal_comparison.angle_degrees is not None
        and normal_comparison.angle_degrees >= minimum_normal_angle_degrees
    )
    corner = check_box_corner(
        matches.first_points[first_homography.inlier_mask],
        matches.first_points[global_mask],
        image_width_px,
        image_height_px,
        roi_width_fraction,
        roi_height_fraction,
        edge_tolerance_px,
        minimum_hull_area_px2,
    )
    orientation_distinct = bool(orientation_distinct)
    qualifies = bool(orientation_distinct and corner.found)
    if not normal_comparison.success:
        reason = normal_comparison.reason or "Plane orientations are ambiguous"
    elif not orientation_distinct:
        reason = "Plane normals are too similar"
    elif not corner.found:
        reason = corner.reason or "Plane regions do not form a box corner"
    else:
        reason = "Two adjacent surfaces with different orientations were detected"

    return SecondPlaneResult(
        True,
        second_model,
        global_mask,
        normal_comparison,
        orientation_distinct,
        corner,
        qualifies,
        reason,
    )


def _completed_model(
    matrix: NDArray[np.floating],
    mask: NDArray[np.integer],
    residuals: FloatArray,
    minimum_inliers: int,
    model_name: str,
) -> GeometryModelResult:
    boolean_mask = np.asarray(mask, dtype=bool).reshape(-1)
    inlier_count = int(boolean_mask.sum())
    point_count = len(boolean_mask)
    success = inlier_count >= minimum_inliers
    return GeometryModelResult(
        success=success,
        matrix=np.asarray(matrix, dtype=np.float64),
        inlier_mask=boolean_mask,
        residuals_squared_px=np.asarray(residuals, dtype=np.float64),
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / point_count if point_count else 0.0,
        reason=(
            None
            if success
            else f"{model_name} has {inlier_count} inliers; {minimum_inliers} required"
        ),
    )


def _failed_model(point_count: int, reason: str) -> GeometryModelResult:
    return GeometryModelResult(
        success=False,
        matrix=None,
        inlier_mask=np.zeros(point_count, dtype=bool),
        residuals_squared_px=np.full(point_count, np.inf, dtype=np.float64),
        inlier_count=0,
        inlier_ratio=0.0,
        reason=reason,
    )


def _validated_point_arrays(first_points, second_points) -> tuple[FloatArray, FloatArray]:
    first = np.asarray(first_points, dtype=np.float64)
    second = np.asarray(second_points, dtype=np.float64)
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape:
        raise ValueError("Point arrays must have matching shape (N, 2)")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Point coordinates must be finite")
    return first, second


def _validated_matrix(matrix, name: str) -> FloatArray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return result


def _perspective_transform(points: FloatArray, matrix: FloatArray) -> FloatArray:
    transformed = cv2.perspectiveTransform(
        points.reshape(-1, 1, 2), matrix
    ).reshape(-1, 2)
    if not np.isfinite(transformed).all():
        raise ValueError("Homography produced non-finite point coordinates")
    return transformed


def _validate_ransac_settings(
    threshold: float,
    confidence: float,
    max_iterations: int,
    minimum_inliers: int,
) -> None:
    if threshold <= 0:
        raise ValueError("RANSAC threshold must be greater than zero")
    if not 0 < confidence < 1:
        raise ValueError("RANSAC confidence must be between 0 and 1")
    if max_iterations < 1:
        raise ValueError("RANSAC max_iterations must be at least 1")
    if minimum_inliers < 1:
        raise ValueError("minimum_inliers must be at least 1")


def _median_movement(matches: FeatureMatches) -> float:
    if not matches.matches:
        return 0.0
    distances = np.linalg.norm(
        matches.second_points.astype(np.float64)
        - matches.first_points.astype(np.float64),
        axis=1,
    )
    return float(np.median(distances))


def _decompose_homography(
    homography: FloatArray,
    camera_matrix: FloatArray,
) -> list[tuple[FloatArray, FloatArray, FloatArray]]:
    try:
        solution_count, rotations, translations, normals = cv2.decomposeHomographyMat(
            homography, camera_matrix
        )
    except cv2.error:
        return []

    solutions = []
    for index in range(solution_count):
        rotation = np.asarray(rotations[index], dtype=np.float64)
        translation = np.asarray(translations[index], dtype=np.float64).reshape(3)
        normal = np.asarray(normals[index], dtype=np.float64).reshape(3)
        if not (
            np.isfinite(rotation).all()
            and np.isfinite(translation).all()
            and np.isfinite(normal).all()
        ):
            continue
        if np.linalg.norm(normal) <= np.finfo(np.float64).eps:
            continue
        solutions.append((rotation, translation, normal / np.linalg.norm(normal)))
    return solutions


def _direction_difference(first: FloatArray, second: FloatArray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    epsilon = np.finfo(np.float64).eps
    if first_norm <= epsilon and second_norm <= epsilon:
        return 0.0
    if first_norm <= epsilon or second_norm <= epsilon:
        return 1.0
    cosine = abs(float(np.dot(first, second) / (first_norm * second_norm)))
    return 1.0 - float(np.clip(cosine, 0.0, 1.0))


def _normalized_vector(vector, name: str) -> FloatArray:
    result = np.asarray(vector, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-dimensional vector")
    norm = float(np.linalg.norm(result))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} cannot be a zero vector")
    return result / norm


def _subset_matches(matches: FeatureMatches, indices: NDArray[np.integer]) -> FeatureMatches:
    selected_indices = np.asarray(indices, dtype=int)
    return FeatureMatches(
        first_frame_index=matches.first_frame_index,
        second_frame_index=matches.second_frame_index,
        matches=[matches.matches[index] for index in selected_indices],
        first_points=matches.first_points[selected_indices].copy(),
        second_points=matches.second_points[selected_indices].copy(),
    )


def _closest_hull_points(
    first_hull: FloatArray,
    second_hull: FloatArray,
) -> tuple[float, FloatArray, FloatArray]:
    intersection_area, intersection = cv2.intersectConvexConvex(
        first_hull.astype(np.float32), second_hull.astype(np.float32)
    )
    if intersection_area > 0 and intersection is not None:
        contact = np.mean(intersection.reshape(-1, 2), axis=0).astype(np.float64)
        return 0.0, contact, contact.copy()

    best_distance = np.inf
    best_first = None
    best_second = None
    for point in first_hull:
        for start, end in _hull_segments(second_hull):
            closest = _closest_point_on_segment(point, start, end)
            distance = float(np.linalg.norm(point - closest))
            if distance < best_distance:
                best_distance, best_first, best_second = distance, point, closest
    for point in second_hull:
        for start, end in _hull_segments(first_hull):
            closest = _closest_point_on_segment(point, start, end)
            distance = float(np.linalg.norm(point - closest))
            if distance < best_distance:
                best_distance, best_first, best_second = distance, closest, point

    return (
        best_distance,
        np.asarray(best_first, dtype=np.float64),
        np.asarray(best_second, dtype=np.float64),
    )


def _hull_segments(hull: FloatArray):
    for index in range(len(hull)):
        yield hull[index], hull[(index + 1) % len(hull)]


def _closest_point_on_segment(
    point: FloatArray,
    start: FloatArray,
    end: FloatArray,
) -> FloatArray:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared <= np.finfo(np.float64).eps:
        return start.copy()
    amount = float(np.dot(point - start, segment) / length_squared)
    return start + np.clip(amount, 0.0, 1.0) * segment


def _point_inside_central_roi(
    point: FloatArray,
    image_width_px: int,
    image_height_px: int,
    width_fraction: float,
    height_fraction: float,
) -> bool:
    roi_width = image_width_px * width_fraction
    roi_height = image_height_px * height_fraction
    left = (image_width_px - roi_width) / 2.0
    right = left + roi_width
    top = (image_height_px - roi_height) / 2.0
    bottom = top + roi_height
    return bool(left <= point[0] <= right and top <= point[1] <= bottom)
