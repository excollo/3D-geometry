"""Shared, validated data types for the 3D box checker."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.uint8]
Descriptors = NDArray[np.floating[Any]] | NDArray[np.uint8]


class Verdict(str, Enum):
    """Outcomes returned by the standalone Part 1 checker."""

    THREE_D = "3d"
    FLAT = "flat"
    INSUFFICIENT = "insufficient"


class GroundTruthLabel(str, Enum):
    """Optional known label used for evaluation, never classification."""

    REAL = "real"
    FLAT_PRINTOUT = "flat_printout"
    FOLDED_PRINTOUT = "folded_printout"
    SCREEN = "screen"
    BAD_RECORDING = "bad_recording"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class AnalysisInput:
    """User-supplied information required to start one analysis."""

    video_path: Path
    label: GroundTruthLabel | None = None
    phone_model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "video_path", Path(self.video_path))

        if isinstance(self.label, str):
            object.__setattr__(self, "label", GroundTruthLabel(self.label))

        if self.phone_model is not None and not self.phone_model.strip():
            raise ValueError("phone_model cannot be blank")


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic properties reported by OpenCV for an opened video file."""

    width_px: int
    height_px: int
    frames_per_second: float
    frame_count: int
    duration_seconds: float

    def __post_init__(self) -> None:
        for name in ("width_px", "height_px", "frames_per_second", "frame_count"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be greater than zero")


@dataclass(slots=True)
class FrameData:
    """One sampled frame and the quality information measured from it."""

    index: int
    timestamp_seconds: float
    original_image: Image
    processed_image: Image
    brightness: float = 0.0
    blur_score: float = 0.0
    usable: bool = True
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative("index", self.index)
        _require_non_negative("timestamp_seconds", self.timestamp_seconds)
        _require_image("original_image", self.original_image)
        _require_image("processed_image", self.processed_image)
        _require_non_negative("brightness", self.brightness)
        _require_non_negative("blur_score", self.blur_score)


@dataclass(slots=True)
class FeatureData:
    """Feature points and descriptors detected in one frame."""

    frame_index: int
    keypoints: list[Any] = field(default_factory=list)
    descriptors: Descriptors | None = None

    def __post_init__(self) -> None:
        _require_non_negative("frame_index", self.frame_index)


@dataclass(slots=True)
class FeatureMatches:
    """Accepted feature correspondences between two different frames."""

    first_frame_index: int
    second_frame_index: int
    matches: list[Any]
    first_points: NDArray[np.float32]
    second_points: NDArray[np.float32]

    def __post_init__(self) -> None:
        _require_non_negative("first_frame_index", self.first_frame_index)
        _require_non_negative("second_frame_index", self.second_frame_index)
        if self.first_frame_index == self.second_frame_index:
            raise ValueError("Feature matches require two different frames")
        for name, points in (
            ("first_points", self.first_points),
            ("second_points", self.second_points),
        ):
            if points.shape != (len(self.matches), 2):
                raise ValueError(
                    f"{name} must have shape ({len(self.matches)}, 2)"
                )


@dataclass(frozen=True, slots=True)
class FramePair:
    """Two sampled frames selected for feature matching."""

    first_frame_index: int
    second_frame_index: int
    time_gap_seconds: float

    def __post_init__(self) -> None:
        _require_non_negative("first_frame_index", self.first_frame_index)
        _require_non_negative("second_frame_index", self.second_frame_index)
        if self.first_frame_index == self.second_frame_index:
            raise ValueError("A frame pair must contain two different frames")
        if self.time_gap_seconds <= 0:
            raise ValueError("time_gap_seconds must be greater than zero")


@dataclass(slots=True)
class PairMetrics:
    """Geometry measurements produced from one pair of video frames."""

    first_frame_index: int
    second_frame_index: int
    time_gap_seconds: float
    match_count: int = 0
    median_movement_px: float = 0.0
    homography_inlier_ratio: float = 0.0
    fundamental_inlier_ratio: float = 0.0
    homography_gric: float | None = None
    fundamental_gric: float | None = None
    median_parallax_px: float = 0.0
    second_plane_found: bool = False
    second_plane_inliers: int = 0
    plane_normal_angle_degrees: float | None = None
    box_corner_found: bool = False
    pair_score: float = 0.0
    low_motion: bool = False
    low_parallax: bool = False
    pure_rotation_risk: bool = False

    def __post_init__(self) -> None:
        for name in (
            "first_frame_index",
            "second_frame_index",
            "time_gap_seconds",
            "match_count",
            "median_movement_px",
            "homography_gric",
            "fundamental_gric",
            "median_parallax_px",
            "second_plane_inliers",
            "plane_normal_angle_degrees",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(name, value)

        for name in (
            "homography_inlier_ratio",
            "fundamental_inlier_ratio",
            "pair_score",
        ):
            _require_unit_interval(name, getattr(self, name))

        if self.first_frame_index == self.second_frame_index:
            raise ValueError("A frame pair must contain two different frames")


@dataclass(slots=True)
class AnalysisMetrics:
    """Summary measurements aggregated across an entire video."""

    frames_sampled: int = 0
    frames_used: int = 0
    pairs_used: int = 0
    median_matches: float = 0.0
    median_movement_px: float = 0.0
    median_h_inlier_ratio: float = 0.0
    median_f_inlier_ratio: float = 0.0
    second_plane_pair_fraction: float = 0.0
    median_parallax_px: float = 0.0
    median_brightness: float = 0.0
    median_blur: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "frames_sampled",
            "frames_used",
            "pairs_used",
            "median_matches",
            "median_movement_px",
            "median_parallax_px",
            "median_brightness",
            "median_blur",
        ):
            _require_non_negative(name, getattr(self, name))

        for name in (
            "median_h_inlier_ratio",
            "median_f_inlier_ratio",
            "second_plane_pair_fraction",
        ):
            _require_unit_interval(name, getattr(self, name))

        if self.frames_used > self.frames_sampled:
            raise ValueError("frames_used cannot exceed frames_sampled")


@dataclass(slots=True)
class AnalysisResult:
    """Final result returned by the standalone checker."""

    verdict: Verdict
    score: float
    confidence: float
    reason: str
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    pairs: list[PairMetrics] = field(default_factory=list)
    processing_ms: int = 0
    label: GroundTruthLabel | None = None
    phone_model: str | None = None
    debug_frames: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.verdict, str):
            self.verdict = Verdict(self.verdict)
        if isinstance(self.label, str):
            self.label = GroundTruthLabel(self.label)

        _require_unit_interval("score", self.score)
        _require_unit_interval("confidence", self.confidence)
        _require_non_negative("processing_ms", self.processing_ms)

        if not self.reason.strip():
            raise ValueError("reason cannot be blank")
        if self.phone_model is not None and not self.phone_model.strip():
            raise ValueError("phone_model cannot be blank")

        self.debug_frames = [Path(path) for path in self.debug_frames]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary for storage and APIs."""
        return _to_json_value(asdict(self))


def _require_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _require_image(name: str, value: Image) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8 pixels")
    if value.ndim not in (2, 3):
        raise ValueError(f"{name} must be a grayscale or colour image")


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    return value
