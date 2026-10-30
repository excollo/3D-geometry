"""Create and save annotated images that explain geometry decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from boxcheck.geometry import PairGeometryResult, SecondPlaneResult
from boxcheck.types import FeatureMatches, FrameData, PairMetrics


Color = tuple[int, int, int]
PLANE_ONE_COLOR: Color = (0, 200, 0)  # green in OpenCV BGR
PLANE_TWO_COLOR: Color = (255, 160, 0)  # blue/cyan in OpenCV BGR
OUTLIER_COLOR: Color = (0, 0, 255)  # red in OpenCV BGR
TEXT_COLOR: Color = (240, 240, 240)
HEADER_HEIGHT_PX = 70


class DebugDrawError(ValueError):
    """Raised when debug drawing input is inconsistent or cannot be saved."""


@dataclass(slots=True)
class DebugPairInput:
    """Everything required to explain one analyzed frame pair visually."""

    first_frame: FrameData
    second_frame: FrameData
    matches: FeatureMatches
    first_plane_mask: NDArray[np.bool_]
    second_plane_mask: NDArray[np.bool_]
    metrics: PairMetrics | None = None


def build_debug_pair_input(
    first_frame: FrameData,
    second_frame: FrameData,
    matches: FeatureMatches,
    geometry: PairGeometryResult,
    second_plane: SecondPlaneResult,
    metrics: PairMetrics | None = None,
) -> DebugPairInput:
    """Adapt geometry results into the masks required for debug drawing."""
    return DebugPairInput(
        first_frame=first_frame,
        second_frame=second_frame,
        matches=matches,
        first_plane_mask=geometry.homography.inlier_mask.copy(),
        second_plane_mask=second_plane.global_inlier_mask.copy(),
        metrics=metrics,
    )


def annotate_pair(debug_input: DebugPairInput) -> NDArray[np.uint8]:
    """Draw classified correspondences on a side-by-side frame canvas."""
    _validate_debug_input(debug_input)
    first_image = _to_bgr(debug_input.first_frame.processed_image)
    second_image = _to_bgr(debug_input.second_frame.processed_image)
    canvas, first_offset, second_offset = _make_canvas(first_image, second_image)

    for index, (first_point, second_point) in enumerate(
        zip(
            debug_input.matches.first_points,
            debug_input.matches.second_points,
            strict=True,
        )
    ):
        color = _match_color(
            debug_input.first_plane_mask[index],
            debug_input.second_plane_mask[index],
        )
        first_position = (
            round(float(first_point[0])) + first_offset[0],
            round(float(first_point[1])) + first_offset[1],
        )
        second_position = (
            round(float(second_point[0])) + second_offset[0],
            round(float(second_point[1])) + second_offset[1],
        )
        cv2.line(canvas, first_position, second_position, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, first_position, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, second_position, 4, color, -1, cv2.LINE_AA)

    _draw_header(canvas, debug_input)
    return canvas


def save_debug_pair_images(
    output_directory: str | Path,
    debug_pairs: list[DebugPairInput],
    maximum_images: int,
    jpeg_quality: int = 90,
) -> list[Path]:
    """Save at most the configured number of annotated pair images."""
    if maximum_images < 0:
        raise ValueError("maximum_images must be non-negative")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if maximum_images == 0 or not debug_pairs:
        return []

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for position, debug_input in enumerate(debug_pairs[:maximum_images]):
        image = annotate_pair(debug_input)
        filename = (
            f"pair_{position:03d}_"
            f"{debug_input.matches.first_frame_index:06d}_"
            f"{debug_input.matches.second_frame_index:06d}.jpg"
        )
        path = directory / filename
        written = cv2.imwrite(
            str(path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not written:
            raise DebugDrawError(f"OpenCV could not save debug image: {path}")
        saved_paths.append(path)

    return saved_paths


def _validate_debug_input(debug_input: DebugPairInput) -> None:
    match_count = len(debug_input.matches.matches)
    if debug_input.first_frame.index != debug_input.matches.first_frame_index:
        raise DebugDrawError("First frame does not match feature-match indices")
    if debug_input.second_frame.index != debug_input.matches.second_frame_index:
        raise DebugDrawError("Second frame does not match feature-match indices")

    first_mask = np.asarray(debug_input.first_plane_mask, dtype=bool).reshape(-1)
    second_mask = np.asarray(debug_input.second_plane_mask, dtype=bool).reshape(-1)
    if len(first_mask) != match_count or len(second_mask) != match_count:
        raise DebugDrawError("Plane masks must contain one value per feature match")
    if np.any(first_mask & second_mask):
        raise DebugDrawError("A match cannot belong to both plane 1 and plane 2")
    if debug_input.metrics is not None and (
        debug_input.metrics.first_frame_index != debug_input.matches.first_frame_index
        or debug_input.metrics.second_frame_index != debug_input.matches.second_frame_index
    ):
        raise DebugDrawError("Pair metrics do not match feature-match indices")


def _to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise DebugDrawError("Debug frames must be grayscale or three-channel BGR images")


def _make_canvas(first_image, second_image):
    first_height, first_width = first_image.shape[:2]
    second_height, second_width = second_image.shape[:2]
    canvas = np.zeros(
        (
            HEADER_HEIGHT_PX + max(first_height, second_height),
            first_width + second_width,
            3,
        ),
        dtype=np.uint8,
    )
    canvas[
        HEADER_HEIGHT_PX : HEADER_HEIGHT_PX + first_height,
        :first_width,
    ] = first_image
    canvas[
        HEADER_HEIGHT_PX : HEADER_HEIGHT_PX + second_height,
        first_width : first_width + second_width,
    ] = second_image
    return canvas, (0, HEADER_HEIGHT_PX), (first_width, HEADER_HEIGHT_PX)


def _match_color(first_plane: bool, second_plane: bool) -> Color:
    if first_plane:
        return PLANE_ONE_COLOR
    if second_plane:
        return PLANE_TWO_COLOR
    return OUTLIER_COLOR


def _draw_header(canvas: NDArray[np.uint8], debug_input: DebugPairInput) -> None:
    cv2.putText(
        canvas,
        f"Frames {debug_input.matches.first_frame_index} -> "
        f"{debug_input.matches.second_frame_index}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    legend = [
        ("Plane 1", PLANE_ONE_COLOR),
        ("Plane 2", PLANE_TWO_COLOR),
        ("Other", OUTLIER_COLOR),
    ]
    x = 10
    for label, color in legend:
        cv2.circle(canvas, (x + 5, 42), 5, color, -1)
        cv2.putText(
            canvas,
            label,
            (x + 14, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        x += 95

    if debug_input.metrics is not None:
        metrics = debug_input.metrics
        text = (
            f"score={metrics.pair_score:.2f}  matches={metrics.match_count}  "
            f"H={metrics.homography_inlier_ratio:.2f}  "
            f"F={metrics.fundamental_inlier_ratio:.2f}  "
            f"parallax={metrics.median_parallax_px:.1f}px"
        )
        cv2.putText(
            canvas,
            text,
            (300, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
