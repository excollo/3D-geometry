"""Open and manage video files used by the checker."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2

from boxcheck.preprocessing import convert_to_grayscale, enhance_contrast_clahe
from boxcheck.types import FrameData, Image, VideoMetadata


class VideoOpenError(ValueError):
    """Raised when a video path cannot be opened by OpenCV."""


@contextmanager
def open_video(video_path: str | Path) -> Iterator[cv2.VideoCapture]:
    """Open a video and always release its OpenCV capture handle.

    The yielded ``VideoCapture`` is ready for metadata queries and frame reads.
    Frame extraction and sampling are implemented separately.
    """
    path = Path(video_path)

    if not path.is_file():
        raise VideoOpenError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoOpenError(f"OpenCV could not open the video file: {path}")

    try:
        yield capture
    finally:
        capture.release()


class VideoMetadataError(ValueError):
    """Raised when OpenCV reports missing or invalid video properties."""


class VideoValidationError(ValueError):
    """Raised when a readable video violates an analysis requirement."""


class VideoReadError(ValueError):
    """Raised when OpenCV cannot decode an expected video frame."""


def read_video_metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    """Read and validate metadata from an opened OpenCV video capture."""
    width_px = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height_px = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames_per_second = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    values = {
        "width": width_px,
        "height": height_px,
        "frame rate": frames_per_second,
        "frame count": frame_count,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        names = ", ".join(invalid)
        raise VideoMetadataError(f"Video has invalid or missing metadata: {names}")

    return VideoMetadata(
        width_px=width_px,
        height_px=height_px,
        frames_per_second=frames_per_second,
        frame_count=frame_count,
        duration_seconds=frame_count / frames_per_second,
    )


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Open a video, read its metadata, and release it."""
    with open_video(video_path) as capture:
        return read_video_metadata(capture)


def validate_video_duration(
    metadata: VideoMetadata,
    maximum_duration_seconds: float,
) -> None:
    """Reject a video whose reported duration exceeds the configured limit."""
    if maximum_duration_seconds <= 0:
        raise ValueError("maximum_duration_seconds must be greater than zero")

    if metadata.duration_seconds > maximum_duration_seconds:
        raise VideoValidationError(
            f"Video is {metadata.duration_seconds:.2f} seconds long; "
            f"the maximum allowed duration is {maximum_duration_seconds:.2f} seconds"
        )


def inspect_video(
    video_path: str | Path,
    maximum_duration_seconds: float,
) -> VideoMetadata:
    """Open a video, read its metadata, and enforce the duration limit."""
    metadata = get_video_metadata(video_path)
    validate_video_duration(metadata, maximum_duration_seconds)
    return metadata


def build_sample_frame_indices(
    metadata: VideoMetadata,
    sample_frames_per_second: float,
) -> list[int]:
    """Choose approximately evenly spaced source-frame indices."""
    if sample_frames_per_second <= 0:
        raise ValueError("sample_frames_per_second must be greater than zero")

    effective_sample_rate = min(
        sample_frames_per_second,
        metadata.frames_per_second,
    )
    source_frames_per_sample = metadata.frames_per_second / effective_sample_rate

    indices: list[int] = []
    position = 0.0
    while position < metadata.frame_count:
        index = min(round(position), metadata.frame_count - 1)
        if not indices or index != indices[-1]:
            indices.append(index)
        position += source_frames_per_sample

    return indices


def sample_frames(
    capture: cv2.VideoCapture,
    metadata: VideoMetadata,
    sample_frames_per_second: float,
    maximum_working_long_side_px: int,
    use_clahe: bool,
    clahe_clip_limit: float,
    clahe_grid_size: int,
) -> list[FrameData]:
    """Decode a video sequentially and retain the selected sample frames."""
    sample_indices = build_sample_frame_indices(metadata, sample_frames_per_second)
    selected_indices = set(sample_indices)
    last_selected_index = sample_indices[-1]
    sampled_frames: list[FrameData] = []

    for frame_index in range(last_selected_index + 1):
        success, image = capture.read()
        if not success or image is None:
            raise VideoReadError(
                f"OpenCV could not decode expected frame {frame_index}"
            )

        if frame_index in selected_indices:
            grayscale_image = convert_to_grayscale(
                resize_frame_to_limit(image, maximum_working_long_side_px)
            )
            working_image = enhance_contrast_clahe(
                grayscale_image,
                enabled=use_clahe,
                clip_limit=clahe_clip_limit,
                grid_size=clahe_grid_size,
            )
            sampled_frames.append(
                FrameData(
                    index=frame_index,
                    timestamp_seconds=frame_index / metadata.frames_per_second,
                    original_image=image,
                    processed_image=working_image,
                )
            )

    return sampled_frames


def extract_sampled_frames(
    video_path: str | Path,
    maximum_duration_seconds: float,
    sample_frames_per_second: float,
    maximum_working_long_side_px: int,
    use_clahe: bool,
    clahe_clip_limit: float,
    clahe_grid_size: int,
) -> tuple[VideoMetadata, list[FrameData]]:
    """Validate a video and return its metadata and sampled frames."""
    with open_video(video_path) as capture:
        metadata = read_video_metadata(capture)
        validate_video_duration(metadata, maximum_duration_seconds)
        frames = sample_frames(
            capture,
            metadata,
            sample_frames_per_second,
            maximum_working_long_side_px,
            use_clahe,
            clahe_clip_limit,
            clahe_grid_size,
        )
    return metadata, frames


def resize_frame_to_limit(image: Image, maximum_long_side_px: int) -> Image:
    """Return a working copy whose longest side does not exceed the limit."""
    # The concrete NumPy image validation is performed by FrameData.
    if maximum_long_side_px <= 0:
        raise ValueError("maximum_long_side_px must be greater than zero")

    height, width = image.shape[:2]
    longest_side = max(width, height)
    if longest_side <= maximum_long_side_px:
        return image.copy()

    scale = maximum_long_side_px / longest_side
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
