"""Tests for useful frame-pair selection."""

import numpy as np
import pytest

from boxcheck.pairing import select_frame_pairs
from boxcheck.types import FrameData


def make_frame(index: int, timestamp: float, usable: bool = True) -> FrameData:
    original = np.zeros((4, 6, 3), dtype=np.uint8)
    processed = np.zeros((4, 6), dtype=np.uint8)
    frame = FrameData(index, timestamp, original, processed)
    frame.usable = usable
    if not usable:
        frame.rejection_reason = "too_blurry"
    return frame


def pair_keys(pairs):
    return {(pair.first_frame_index, pair.second_frame_index) for pair in pairs}


def test_pairs_below_minimum_gap_are_excluded() -> None:
    frames = [make_frame(0, 0.0), make_frame(1, 0.59)]

    assert select_frame_pairs(frames, 0.60, 1.50, 20) == []


def test_pair_at_exact_minimum_gap_is_accepted() -> None:
    frames = [make_frame(0, 0.0), make_frame(1, 0.60)]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 20)

    assert pair_keys(pairs) == {(0, 1)}
    assert pairs[0].time_gap_seconds == pytest.approx(0.60)


def test_pair_at_exact_maximum_gap_is_accepted() -> None:
    frames = [make_frame(0, 0.0), make_frame(1, 1.50)]

    assert pair_keys(select_frame_pairs(frames, 0.60, 1.50, 20)) == {(0, 1)}


def test_pairs_above_maximum_gap_are_excluded() -> None:
    frames = [make_frame(0, 0.0), make_frame(1, 1.51)]

    assert select_frame_pairs(frames, 0.60, 1.50, 20) == []


def test_rejected_frames_are_ignored() -> None:
    frames = [
        make_frame(0, 0.0),
        make_frame(1, 0.75, usable=False),
        make_frame(2, 1.0),
    ]

    assert pair_keys(select_frame_pairs(frames, 0.60, 1.50, 20)) == {(0, 2)}


def test_unsorted_frames_are_sorted_by_timestamp() -> None:
    frames = [make_frame(2, 1.0), make_frame(0, 0.0), make_frame(1, 0.75)]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 20)

    assert all(pair.first_frame_index != 2 for pair in pairs)
    assert (0, 2) in pair_keys(pairs)


def test_frame_is_never_paired_with_itself() -> None:
    frames = [make_frame(0, 0.0), make_frame(1, 1.0)]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 20)

    assert all(pair.first_frame_index != pair.second_frame_index for pair in pairs)


def test_reversed_duplicate_pairs_are_not_created() -> None:
    frames = [make_frame(index, index * 0.25) for index in range(8)]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 20)
    keys = pair_keys(pairs)

    assert len(keys) == len(pairs)
    assert all((second, first) not in keys for first, second in keys)


def test_pair_count_is_limited() -> None:
    frames = [make_frame(index, index * 0.25) for index in range(40)]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 20)

    assert len(pairs) == 20


def test_limited_pairs_are_distributed_across_video() -> None:
    frames = [make_frame(index, index * 0.25) for index in range(40)]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 5)
    first_indices = [pair.first_frame_index for pair in pairs]

    assert len(pairs) == 5
    assert min(first_indices) <= 1
    assert max(first_indices) >= 33


def test_pair_nearest_middle_of_gap_range_is_preferred() -> None:
    frames = [
        make_frame(0, 0.0),
        make_frame(1, 0.60),
        make_frame(2, 1.00),
        make_frame(3, 1.50),
    ]

    pairs = select_frame_pairs(frames, 0.60, 1.50, 1)

    assert pair_keys(pairs) == {(0, 2)}


def test_empty_or_single_frame_input_returns_no_pairs() -> None:
    assert select_frame_pairs([], 0.60, 1.50, 20) == []
    assert select_frame_pairs([make_frame(0, 0.0)], 0.60, 1.50, 20) == []


@pytest.mark.parametrize(
    ("minimum_gap", "maximum_gap", "maximum_pairs", "message"),
    [
        (0.0, 1.50, 20, "minimum_gap_seconds"),
        (1.50, 0.60, 20, "maximum_gap_seconds"),
        (0.60, 1.50, 0, "maximum_pairs"),
    ],
)
def test_invalid_pairing_settings_are_rejected(
    minimum_gap: float,
    maximum_gap: float,
    maximum_pairs: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_frame_pairs([], minimum_gap, maximum_gap, maximum_pairs)


def test_duplicate_usable_frame_indices_are_rejected() -> None:
    frames = [make_frame(0, 0.0), make_frame(0, 1.0)]

    with pytest.raises(ValueError, match="unique frame indices"):
        select_frame_pairs(frames, 0.60, 1.50, 20)
