"""Tests for safely opening video files with OpenCV."""

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from boxcheck.video import (
    VideoMetadataError,
    VideoOpenError,
    VideoReadError,
    VideoValidationError,
    build_sample_frame_indices,
    extract_sampled_frames,
    get_video_metadata,
    inspect_video,
    open_video,
    read_video_metadata,
    resize_frame_to_limit,
    sample_frames,
    validate_video_duration,
)
from boxcheck.types import VideoMetadata


def test_open_video_rejects_a_missing_file(tmp_path: Path) -> None:
    missing_video = tmp_path / "missing.mp4"

    with pytest.raises(VideoOpenError, match="Video file not found"):
        with open_video(missing_video):
            pass


def test_open_video_yields_and_releases_capture(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    capture = Mock()
    capture.isOpened.return_value = True

    with patch("boxcheck.video.cv2.VideoCapture", return_value=capture) as constructor:
        with open_video(video_path) as opened_capture:
            assert opened_capture is capture
            capture.release.assert_not_called()

    constructor.assert_called_once_with(str(video_path))
    capture.release.assert_called_once_with()


def test_read_video_metadata_calculates_duration() -> None:
    capture = Mock()
    properties = {
        3: 1280.0,  # cv2.CAP_PROP_FRAME_WIDTH
        4: 720.0,  # cv2.CAP_PROP_FRAME_HEIGHT
        5: 30.0,  # cv2.CAP_PROP_FPS
        7: 240.0,  # cv2.CAP_PROP_FRAME_COUNT
    }
    capture.get.side_effect = properties.__getitem__

    metadata = read_video_metadata(capture)

    assert metadata.width_px == 1280
    assert metadata.height_px == 720
    assert metadata.frames_per_second == 30.0
    assert metadata.frame_count == 240
    assert metadata.duration_seconds == 8.0


def test_read_video_metadata_rejects_invalid_frame_rate() -> None:
    capture = Mock()
    properties = {3: 1280.0, 4: 720.0, 5: 0.0, 7: 240.0}
    capture.get.side_effect = properties.__getitem__

    with pytest.raises(VideoMetadataError, match="frame rate"):
        read_video_metadata(capture)


def test_get_video_metadata_opens_and_releases_video(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    properties = {3: 1920.0, 4: 1080.0, 5: 25.0, 7: 200.0}
    capture.get.side_effect = properties.__getitem__

    with patch("boxcheck.video.cv2.VideoCapture", return_value=capture):
        metadata = get_video_metadata(video_path)

    assert metadata.duration_seconds == 8.0
    capture.release.assert_called_once_with()


def test_validate_video_duration_accepts_video_at_limit() -> None:
    metadata = VideoMetadata(1280, 720, 30.0, 360, 12.0)

    validate_video_duration(metadata, maximum_duration_seconds=12.0)


def test_validate_video_duration_rejects_video_over_limit() -> None:
    metadata = VideoMetadata(1280, 720, 30.0, 361, 12.03)

    with pytest.raises(VideoValidationError, match="maximum allowed duration"):
        validate_video_duration(metadata, maximum_duration_seconds=12.0)


def test_inspect_video_reads_and_validates_in_one_call(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    properties = {3: 1280.0, 4: 720.0, 5: 30.0, 7: 240.0}
    capture.get.side_effect = properties.__getitem__

    with patch("boxcheck.video.cv2.VideoCapture", return_value=capture):
        metadata = inspect_video(video_path, maximum_duration_seconds=12.0)

    assert metadata.duration_seconds == 8.0
    capture.release.assert_called_once_with()


def test_duration_limit_comes_from_threshold_configuration() -> None:
    from boxcheck.settings import load_thresholds

    maximum_duration = load_thresholds()["video"]["maximum_duration_seconds"]

    assert maximum_duration == 12.0


def test_build_sample_indices_selects_about_four_frames_per_second() -> None:
    metadata = VideoMetadata(1280, 720, 30.0, 240, 8.0)

    indices = build_sample_frame_indices(metadata, sample_frames_per_second=4.0)

    assert len(indices) == 32
    assert indices[:4] == [0, 8, 15, 22]
    assert indices == sorted(set(indices))


def test_sampling_does_not_duplicate_frames_for_low_fps_video() -> None:
    metadata = VideoMetadata(640, 480, 2.0, 4, 2.0)

    indices = build_sample_frame_indices(metadata, sample_frames_per_second=4.0)

    assert indices == [0, 1, 2, 3]


def test_sample_frames_preserves_source_index_and_timestamp() -> None:
    metadata = VideoMetadata(4, 3, 8.0, 8, 1.0)
    capture = Mock()
    images = [np.full((3, 4, 3), index, dtype=np.uint8) for index in range(8)]
    capture.read.side_effect = [(True, image) for image in images]

    frames = sample_frames(
        capture,
        metadata,
        sample_frames_per_second=4.0,
        maximum_working_long_side_px=1080,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_grid_size=8,
    )

    assert [frame.index for frame in frames] == [0, 2, 4, 6]
    assert [frame.timestamp_seconds for frame in frames] == [0.0, 0.25, 0.5, 0.75]
    assert [int(frame.original_image[0, 0, 0]) for frame in frames] == [0, 2, 4, 6]


def test_sample_frames_reports_decode_failure() -> None:
    metadata = VideoMetadata(4, 3, 8.0, 8, 1.0)
    capture = Mock()
    capture.read.return_value = (False, None)

    with pytest.raises(VideoReadError, match="expected frame 0"):
        sample_frames(
            capture,
            metadata,
            sample_frames_per_second=4.0,
            maximum_working_long_side_px=1080,
            use_clahe=True,
            clahe_clip_limit=2.0,
            clahe_grid_size=8,
        )


def test_extract_sampled_frames_uses_configured_rates(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    properties = {3: 4.0, 4: 3.0, 5: 8.0, 7: 8.0}
    capture.get.side_effect = properties.__getitem__
    images = [np.zeros((3, 4, 3), dtype=np.uint8) for _ in range(8)]
    capture.read.side_effect = [(True, image) for image in images]

    with patch("boxcheck.video.cv2.VideoCapture", return_value=capture):
        metadata, frames = extract_sampled_frames(
            video_path,
            maximum_duration_seconds=12.0,
            sample_frames_per_second=4.0,
            maximum_working_long_side_px=1080,
            use_clahe=True,
            clahe_clip_limit=2.0,
            clahe_grid_size=8,
        )

    assert metadata.duration_seconds == 1.0
    assert len(frames) == 4
    capture.release.assert_called_once_with()


def test_resize_frame_downscales_landscape_image_and_preserves_aspect_ratio() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    resized = resize_frame_to_limit(image, maximum_long_side_px=1080)

    assert resized.shape == (608, 1080, 3)


def test_resize_frame_downscales_portrait_image() -> None:
    image = np.zeros((1920, 1080, 3), dtype=np.uint8)

    resized = resize_frame_to_limit(image, maximum_long_side_px=1080)

    assert resized.shape == (1080, 608, 3)


def test_resize_frame_does_not_enlarge_small_image() -> None:
    image = np.zeros((540, 960, 3), dtype=np.uint8)

    resized = resize_frame_to_limit(image, maximum_long_side_px=1080)

    assert resized.shape == image.shape
    assert resized is not image


def test_sample_frames_keeps_original_and_resizes_working_image() -> None:
    metadata = VideoMetadata(1920, 1080, 1.0, 1, 1.0)
    capture = Mock()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    capture.read.return_value = (True, image)

    frames = sample_frames(
        capture,
        metadata,
        sample_frames_per_second=1.0,
        maximum_working_long_side_px=1080,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_grid_size=8,
    )

    assert frames[0].original_image.shape == (1080, 1920, 3)
    assert frames[0].processed_image.shape == (608, 1080)


def test_complete_extraction_with_a_real_synthetic_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        8.0,
        (64, 48),
    )
    assert writer.isOpened(), "OpenCV could not create the synthetic test video"

    try:
        for frame_index in range(8):
            frame = np.full((48, 64, 3), 40 + frame_index * 20, dtype=np.uint8)
            cv2.putText(
                frame,
                str(frame_index),
                (8, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()

    metadata, frames = extract_sampled_frames(
        video_path,
        maximum_duration_seconds=12.0,
        sample_frames_per_second=4.0,
        maximum_working_long_side_px=1080,
        use_clahe=True,
        clahe_clip_limit=2.0,
        clahe_grid_size=8,
    )

    assert metadata.width_px == 64
    assert metadata.height_px == 48
    assert metadata.frames_per_second == pytest.approx(8.0)
    assert metadata.duration_seconds == pytest.approx(1.0)
    assert [frame.index for frame in frames] == [0, 2, 4, 6]
    assert all(frame.original_image.shape == (48, 64, 3) for frame in frames)
    assert all(frame.processed_image.shape == (48, 64) for frame in frames)
    assert all(frame.processed_image.dtype == np.uint8 for frame in frames)


def test_open_video_releases_capture_when_opencv_cannot_open_file(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "invalid.mp4"
    video_path.touch()
    capture = Mock()
    capture.isOpened.return_value = False

    with patch("boxcheck.video.cv2.VideoCapture", return_value=capture):
        with pytest.raises(VideoOpenError, match="OpenCV could not open"):
            with open_video(video_path):
                pass

    capture.release.assert_called_once_with()
