"""Tests for image preprocessing operations."""

import numpy as np
import pytest

from boxcheck.preprocessing import (
    PreprocessingError,
    convert_to_grayscale,
    enhance_contrast_clahe,
)


def test_convert_bgr_image_to_grayscale() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    grayscale = convert_to_grayscale(image)

    assert grayscale.shape == (4, 6)
    assert grayscale.dtype == np.uint8
    assert np.all(grayscale == 76)


def test_convert_to_grayscale_copies_an_already_grayscale_image() -> None:
    image = np.full((4, 6), 100, dtype=np.uint8)

    grayscale = convert_to_grayscale(image)

    assert np.array_equal(grayscale, image)
    assert grayscale is not image


def test_convert_to_grayscale_rejects_unexpected_channel_count() -> None:
    image = np.zeros((4, 6, 4), dtype=np.uint8)

    with pytest.raises(PreprocessingError, match="three-channel BGR"):
        convert_to_grayscale(image)


def test_clahe_enhances_a_low_contrast_grayscale_image() -> None:
    image = np.tile(np.arange(100, 116, dtype=np.uint8), (64, 4))

    enhanced = enhance_contrast_clahe(
        image,
        enabled=True,
        clip_limit=2.0,
        grid_size=8,
    )

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
    assert not np.array_equal(enhanced, image)


def test_disabled_clahe_returns_an_unchanged_copy() -> None:
    image = np.full((16, 16), 100, dtype=np.uint8)

    enhanced = enhance_contrast_clahe(
        image,
        enabled=False,
        clip_limit=2.0,
        grid_size=8,
    )

    assert np.array_equal(enhanced, image)
    assert enhanced is not image


def test_clahe_rejects_a_colour_image() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(PreprocessingError, match="grayscale"):
        enhance_contrast_clahe(image, True, 2.0, 8)
