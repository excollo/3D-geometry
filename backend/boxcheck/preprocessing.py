"""Image preprocessing operations used before feature analysis."""

from __future__ import annotations

import cv2

from boxcheck.types import Image


class PreprocessingError(ValueError):
    """Raised when an image cannot be preprocessed safely."""


def convert_to_grayscale(image: Image) -> Image:
    """Return a grayscale copy of a BGR or already-grayscale image."""
    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise PreprocessingError(
        "Expected a grayscale image or a three-channel BGR image"
    )


def enhance_contrast_clahe(
    grayscale_image: Image,
    enabled: bool,
    clip_limit: float,
    grid_size: int,
) -> Image:
    """Enhance local contrast in a grayscale image using CLAHE."""
    if grayscale_image.ndim != 2:
        raise PreprocessingError("CLAHE requires a grayscale image")
    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than zero")
    if grid_size <= 0:
        raise ValueError("grid_size must be greater than zero")
    if not enabled:
        return grayscale_image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    return clahe.apply(grayscale_image)
