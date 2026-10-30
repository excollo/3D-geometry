"""Select useful, time-separated sampled-frame pairs."""

from __future__ import annotations

from collections import defaultdict

from boxcheck.types import FrameData, FramePair


def select_frame_pairs(
    frames: list[FrameData],
    minimum_gap_seconds: float,
    maximum_gap_seconds: float,
    maximum_pairs: int,
) -> list[FramePair]:
    """Select well-spaced pairs from usable frames across the whole video."""
    _validate_pairing_settings(
        minimum_gap_seconds,
        maximum_gap_seconds,
        maximum_pairs,
    )

    usable_frames = sorted(
        (frame for frame in frames if frame.usable),
        key=lambda frame: (frame.timestamp_seconds, frame.index),
    )
    _require_unique_frame_indices(usable_frames)
    if len(usable_frames) < 2:
        return []

    target_gap = (minimum_gap_seconds + maximum_gap_seconds) / 2.0
    candidates_by_first: dict[int, list[FramePair]] = defaultdict(list)
    epsilon = 1e-9

    for position, first in enumerate(usable_frames[:-1]):
        for second in usable_frames[position + 1 :]:
            gap = second.timestamp_seconds - first.timestamp_seconds
            if gap < minimum_gap_seconds - epsilon:
                continue
            if gap > maximum_gap_seconds + epsilon:
                break

            candidates_by_first[first.index].append(
                FramePair(
                    first_frame_index=first.index,
                    second_frame_index=second.index,
                    time_gap_seconds=gap,
                )
            )

    if not candidates_by_first:
        return []

    for candidates in candidates_by_first.values():
        candidates.sort(
            key=lambda pair: (
                abs(pair.time_gap_seconds - target_gap),
                pair.second_frame_index,
            )
        )

    ordered_first_indices = [
        frame.index for frame in usable_frames if frame.index in candidates_by_first
    ]
    selected: list[FramePair] = []
    candidate_rank = 0

    while len(selected) < maximum_pairs:
        available_first_indices = [
            frame_index
            for frame_index in ordered_first_indices
            if candidate_rank < len(candidates_by_first[frame_index])
        ]
        if not available_first_indices:
            break

        remaining_slots = maximum_pairs - len(selected)
        selection_count = min(remaining_slots, len(available_first_indices))
        if selection_count == 1:
            selected_first_indices = [
                min(
                    available_first_indices,
                    key=lambda frame_index: (
                        abs(
                            candidates_by_first[frame_index][candidate_rank].time_gap_seconds
                            - target_gap
                        ),
                        ordered_first_indices.index(frame_index),
                    ),
                )
            ]
        else:
            selected_first_indices = _spread_across_video(
                available_first_indices,
                selection_count,
            )
        selected.extend(
            candidates_by_first[frame_index][candidate_rank]
            for frame_index in selected_first_indices
        )
        candidate_rank += 1

    frame_order = {frame.index: position for position, frame in enumerate(usable_frames)}
    selected.sort(
        key=lambda pair: (
            frame_order[pair.first_frame_index],
            frame_order[pair.second_frame_index],
        )
    )
    return selected


def _spread_across_video(values: list[int], count: int) -> list[int]:
    """Choose ordered values at approximately even positions."""
    if count >= len(values):
        return values.copy()
    if count == 1:
        return [values[len(values) // 2]]

    positions = [
        round(index * (len(values) - 1) / (count - 1)) for index in range(count)
    ]
    return [values[position] for position in positions]


def _validate_pairing_settings(
    minimum_gap_seconds: float,
    maximum_gap_seconds: float,
    maximum_pairs: int,
) -> None:
    if minimum_gap_seconds <= 0:
        raise ValueError("minimum_gap_seconds must be greater than zero")
    if maximum_gap_seconds < minimum_gap_seconds:
        raise ValueError(
            "maximum_gap_seconds must be greater than or equal to minimum_gap_seconds"
        )
    if maximum_pairs < 1:
        raise ValueError("maximum_pairs must be at least 1")


def _require_unique_frame_indices(frames: list[FrameData]) -> None:
    indices = [frame.index for frame in frames]
    if len(indices) != len(set(indices)):
        raise ValueError("Usable frames must have unique frame indices")
