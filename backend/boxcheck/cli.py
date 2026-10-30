"""Command-line interface for analyzing one video file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from boxcheck.pipeline import analyze_video
from boxcheck.settings import SettingsError
from boxcheck.video import VideoMetadataError, VideoOpenError, VideoReadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boxcheck",
        description="Analyze whether a video behaves like a physical 3D box or a flat image.",
    )
    parser.add_argument("video", type=Path, help="Path to the video file")
    parser.add_argument(
        "--label",
        choices=[
            "real",
            "flat_printout",
            "folded_printout",
            "screen",
            "bad_recording",
            "other",
        ],
        help="Optional known label used only for evaluation",
    )
    parser.add_argument("--phone-model", help="Optional recording phone model")
    parser.add_argument("--debug-dir", type=Path, help="Directory for annotated JPEGs")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    try:
        result = analyze_video(
            args.video,
            label=args.label,
            phone_model=args.phone_model,
            debug_output_directory=args.debug_dir,
        )
    except (VideoOpenError, VideoMetadataError, VideoReadError, SettingsError, ValueError) as error:
        print(f"boxcheck: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
