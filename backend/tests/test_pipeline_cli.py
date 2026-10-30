"""End-to-end tests using real temporary sample video files."""

import json
from pathlib import Path

import cv2
import numpy as np

from boxcheck.cli import main
from boxcheck.pipeline import analyze_video
from boxcheck.types import Verdict


def write_video(path: Path, frames: list[np.ndarray], fps: float = 8.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened(), "OpenCV could not create the sample video"
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def create_flat_sample_video(path: Path) -> None:
    random = np.random.default_rng(404)
    base = np.full((240, 320, 3), 110, dtype=np.uint8)
    for x, y in random.integers([35, 35], [285, 205], size=(140, 2)):
        color = int(random.integers(40, 230))
        cv2.circle(base, (int(x), int(y)), 2, (color, color, color), -1)
    cv2.putText(
        base,
        "IPHONE BOX 123",
        (45, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    frames = []
    for index in range(16):
        transform = np.float32([[1, 0, index * 2.0], [0, 1, index * 0.35]])
        frame = cv2.warpAffine(
            base,
            transform,
            (320, 240),
            borderMode=cv2.BORDER_REFLECT,
        )
        frames.append(frame)
    write_video(path, frames)


def create_bad_sample_video(path: Path) -> None:
    frames = [np.full((240, 320, 3), 100, dtype=np.uint8) for _ in range(16)]
    write_video(path, frames)


def test_complete_pipeline_classifies_real_flat_sample_video(tmp_path: Path) -> None:
    video = tmp_path / "flat_sample.avi"
    debug_directory = tmp_path / "debug"
    create_flat_sample_video(video)

    result = analyze_video(
        video,
        label="flat_printout",
        phone_model="Synthetic Test Camera",
        debug_output_directory=debug_directory,
    )

    assert result.verdict is Verdict.FLAT
    assert result.metrics.frames_sampled == 8
    assert result.metrics.frames_used == 8
    assert result.metrics.pairs_used >= 3
    assert result.metrics.median_movement_px >= 8.0
    assert result.debug_frames
    assert len(result.debug_frames) <= 5
    assert all(path.is_file() for path in result.debug_frames)


def test_complete_pipeline_returns_insufficient_for_bad_sample_video(
    tmp_path: Path,
) -> None:
    video = tmp_path / "bad_sample.avi"
    create_bad_sample_video(video)

    result = analyze_video(video, label="bad_recording")

    assert result.verdict is Verdict.INSUFFICIENT
    assert result.metrics.frames_sampled == 8
    assert result.metrics.frames_used == 0
    assert "blurry" in result.reason.lower() or "detail" in result.reason.lower()


def test_cli_prints_json_result_for_sample_video(
    tmp_path: Path,
    capsys,
) -> None:
    video = tmp_path / "bad_sample.avi"
    create_bad_sample_video(video)

    exit_code = main([str(video), "--label", "bad_recording", "--indent", "0"])
    output = capsys.readouterr()
    data = json.loads(output.out)

    assert exit_code == 0
    assert output.err == ""
    assert data["verdict"] == "insufficient"
    assert data["label"] == "bad_recording"
    assert data["metrics"]["frames_sampled"] == 8


def test_cli_reports_missing_video_as_command_error(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.avi"

    exit_code = main([str(missing)])
    output = capsys.readouterr()

    assert exit_code == 2
    assert output.out == ""
    assert "Video file not found" in output.err
