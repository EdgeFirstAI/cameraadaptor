"""Utilities for EdgeFirst HAL integration.

This module provides helper functions to bridge CameraAdaptor color spaces
with EdgeFirst HAL FourCC codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .color_spaces import SUPPORTED_COLOR_SPACES, ColorSpace

if TYPE_CHECKING:
    import edgefirst_hal  # type: ignore[import-not-found]


# Mapping from ColorSpace to HAL FourCC string codes (packed/interleaved)
# These correspond to V4L2/HAL FourCC conventions
FOURCC_MAP: dict[ColorSpace, str] = {
    ColorSpace.RGB: "RGB3",
    ColorSpace.BGR: "BGR3",
    ColorSpace.RGBA: "RGBA",
    ColorSpace.BGRA: "BGRA",
    ColorSpace.GREY: "GREY",
    ColorSpace.YUYV: "YUYV",
    # Future: Add these when supported in CameraAdaptor
    # ColorSpace.NV12: "NV12",
    # ColorSpace.NV21: "NV21",
}

# Planar format FourCC codes (channels_first / NCHW memory layout)
# These map ColorSpace to their planar FourCC equivalents
PLANAR_FOURCC_MAP: dict[ColorSpace, str] = {
    ColorSpace.RGB: "RGBP",  # Planar RGB (R, G, B planes)
    ColorSpace.RGBA: "RGBP",  # Planar RGBA (alpha ignored/separate)
    ColorSpace.BGR: "BGRP",  # Planar BGR
    ColorSpace.BGRA: "BGRP",  # Planar BGRA (alpha ignored/separate)
}

# Additional HAL FourCC codes for validator migration
# HAL-specific formats without direct ColorSpace equivalents
HAL_FOURCC_ALIASES: dict[str, str] = {
    # Planar format aliases
    "PLANAR_RGB": "RGBP",
    "PLANAR_RGBA": "RGBP",
    "PLANAR_BGR": "BGRP",
    "PLANAR_BGRA": "BGRP",
    # YUV semi-planar formats
    "NV16": "NV16",  # YUV 4:2:2 semi-planar (Y plane + interleaved UV)
    "NV12": "NV12",  # YUV 4:2:0 semi-planar (roadmap)
    "NV21": "NV21",  # YUV 4:2:0 semi-planar VU order (roadmap)
}

# Reverse mapping from FourCC string to ColorSpace (packed formats only)
COLORSPACE_MAP: dict[str, ColorSpace] = {v: k for k, v in FOURCC_MAP.items()}


def get_fourcc_str(color_space: str | ColorSpace) -> str:
    """Get the FourCC string code for a CameraAdaptor color space.

    This returns the standard FourCC string identifier that can be used
    with various video/camera APIs.

    Args:
        color_space: CameraAdaptor color space as string or ColorSpace enum.

    Returns:
        FourCC string code (e.g., "YUYV", "BGR3", "RGBA").

    Raises:
        ValueError: If the color space is not supported or has no
            FourCC mapping.

    Example:
        >>> get_fourcc_str("yuyv")
        'YUYV'
        >>> get_fourcc_str(ColorSpace.BGR)
        'BGR3'
    """
    if isinstance(color_space, str):
        try:
            color_space = ColorSpace(color_space.lower())
        except ValueError:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Invalid color space: {color_space}. "
                f"Supported: {supported}"
            ) from None

    if color_space not in FOURCC_MAP:
        raise ValueError(
            f"No FourCC mapping for color space '{color_space.value}'. "
            f"Supported: {', '.join(cs.value for cs in FOURCC_MAP)}"
        )

    return FOURCC_MAP[color_space]


def get_fourcc(color_space: str | ColorSpace) -> edgefirst_hal.FourCC:
    """Get the HAL FourCC enum for a CameraAdaptor color space.

    This function requires the edgefirst_hal package to be installed.

    Args:
        color_space: CameraAdaptor color space as string or ColorSpace enum.

    Returns:
        edgefirst_hal.FourCC enum value.

    Raises:
        ImportError: If edgefirst_hal is not installed.
        ValueError: If color space is not supported or has no mapping.

    Example:
        >>> fourcc = get_fourcc("yuyv")
        >>> hal_pipeline.set_output_format(fourcc)
    """
    try:
        import edgefirst_hal
    except ImportError:
        raise ImportError(
            "edgefirst_hal is required for FourCC conversion. "
            "Install with: pip install edgefirst-hal"
        ) from None

    fourcc_str = get_fourcc_str(color_space)

    try:
        return getattr(edgefirst_hal.FourCC, fourcc_str)
    except AttributeError:
        raise ValueError(
            f"FourCC '{fourcc_str}' not found in edgefirst_hal.FourCC. "
            "This may indicate a version mismatch."
        ) from None


def from_fourcc_str(fourcc: str) -> ColorSpace:
    """Get the CameraAdaptor ColorSpace for a FourCC string code.

    Args:
        fourcc: FourCC string code (e.g., "YUYV", "BGR3").

    Returns:
        ColorSpace enum value.

    Raises:
        ValueError: If the FourCC code is not recognized.

    Example:
        >>> from_fourcc_str("YUYV")
        <ColorSpace.YUYV: 'yuyv'>
    """
    fourcc_upper = fourcc.upper()
    if fourcc_upper not in COLORSPACE_MAP:
        raise ValueError(
            f"Unknown FourCC code: {fourcc}. "
            f"Supported: {', '.join(COLORSPACE_MAP.keys())}"
        )
    return COLORSPACE_MAP[fourcc_upper]


def from_fourcc(fourcc: edgefirst_hal.FourCC) -> ColorSpace:
    """Get the CameraAdaptor ColorSpace for a HAL FourCC enum.

    This function requires the edgefirst_hal package to be installed.

    Args:
        fourcc: edgefirst_hal.FourCC enum value.

    Returns:
        ColorSpace enum value.

    Raises:
        ValueError: If the FourCC code is not recognized.

    Example:
        >>> import edgefirst_hal as hal
        >>> from_fourcc(hal.FourCC.YUYV)
        <ColorSpace.YUYV: 'yuyv'>
    """
    return from_fourcc_str(fourcc.name)


def get_planar_fourcc_str(color_space: str | ColorSpace) -> str:
    """Get the planar FourCC string code for a CameraAdaptor color space.

    Planar formats store each channel in a separate contiguous memory region,
    corresponding to channels_first (NCHW) memory layout.

    Args:
        color_space: CameraAdaptor color space as string or ColorSpace enum.

    Returns:
        Planar FourCC string code (e.g., "RGBP", "BGRP").

    Raises:
        ValueError: If the color space has no planar format mapping.

    Example:
        >>> get_planar_fourcc_str("rgb")
        'RGBP'
        >>> get_planar_fourcc_str(ColorSpace.BGR)
        'BGRP'
    """
    if isinstance(color_space, str):
        try:
            color_space = ColorSpace(color_space.lower())
        except ValueError:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Invalid color space: {color_space}. "
                f"Supported: {supported}"
            ) from None

    if color_space not in PLANAR_FOURCC_MAP:
        raise ValueError(
            f"No planar FourCC mapping for color space '{color_space.value}'. "
            f"Supported: {', '.join(cs.value for cs in PLANAR_FOURCC_MAP)}"
        )

    return PLANAR_FOURCC_MAP[color_space]


def resolve_hal_alias(alias: str) -> str:
    """Resolve a HAL FourCC alias to its canonical FourCC code.

    This function handles HAL-specific format names that may not directly
    correspond to CameraAdaptor ColorSpace values but are needed for
    validator migration and HAL compatibility.

    Args:
        alias: HAL FourCC alias (e.g., "PLANAR_RGB", "NV16").

    Returns:
        Canonical FourCC string code.

    Raises:
        ValueError: If the alias is not recognized.

    Example:
        >>> resolve_hal_alias("PLANAR_RGB")
        'RGBP'
        >>> resolve_hal_alias("NV16")
        'NV16'
    """
    alias_upper = alias.upper()
    if alias_upper in HAL_FOURCC_ALIASES:
        return HAL_FOURCC_ALIASES[alias_upper]

    # Check if it's already a valid FourCC code
    if alias_upper in COLORSPACE_MAP:
        return alias_upper

    raise ValueError(
        f"Unknown HAL FourCC alias: {alias}. "
        f"Known aliases: {', '.join(HAL_FOURCC_ALIASES.keys())}"
    )


__all__ = [
    "FOURCC_MAP",
    "PLANAR_FOURCC_MAP",
    "HAL_FOURCC_ALIASES",
    "COLORSPACE_MAP",
    "get_fourcc_str",
    "get_fourcc",
    "get_planar_fourcc_str",
    "from_fourcc_str",
    "from_fourcc",
    "resolve_hal_alias",
]
