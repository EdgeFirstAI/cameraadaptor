"""Color space definitions and utilities for camera adaptor."""

from enum import Enum


class ColorSpace(Enum):
    """Supported color space formats for camera adaptor.

    RGB/BGR Family:
        RGB: Standard RGB format (3 channels)
        BGR: OpenCV-native BGR format (3 channels)
        RGBA: RGB + alpha channel (4 input channels, 3 output)
        BGRA: BGR + alpha channel (4 input channels, 3 output)

    Grayscale:
        GREY: Single-channel grayscale (1 channel)

    YUV Family (for platforms with limited ISP):
        YUYV: YUV 4:2:2 packed format (2 channels)
        NV12: YUV 4:2:0 semi-planar (roadmap)
        NV21: YUV 4:2:0 semi-planar Android default (roadmap)

    Bayer Family (for raw camera/no ISP scenarios, roadmap):
        RGGB: Bayer pattern with R at (0,0)
        BGGR: Bayer pattern with B at (0,0)
        GRBG: Bayer pattern with G at (0,0), R at (0,1)
        GBRG: Bayer pattern with G at (0,0), B at (0,1)
    """

    # RGB/BGR Family
    RGB = "rgb"
    BGR = "bgr"
    RGBA = "rgba"
    BGRA = "bgra"

    # Grayscale
    GREY = "grey"

    # YUV Family
    YUYV = "yuyv"
    NV12 = "nv12"
    NV21 = "nv21"

    # Bayer Family (roadmap)
    RGGB = "rggb"
    BGGR = "bggr"
    GRBG = "grbg"
    GBRG = "gbrg"


# Channel counts for input (what the model receives)
_INPUT_CHANNELS = {
    ColorSpace.RGB: 3,
    ColorSpace.BGR: 3,
    ColorSpace.RGBA: 4,
    ColorSpace.BGRA: 4,
    ColorSpace.GREY: 1,
    ColorSpace.YUYV: 2,
    ColorSpace.NV12: 1,  # Planar format, treated as single channel
    ColorSpace.NV21: 1,  # Planar format, treated as single channel
    ColorSpace.RGGB: 1,
    ColorSpace.BGGR: 1,
    ColorSpace.GRBG: 1,
    ColorSpace.GBRG: 1,
}

# Channel counts for output (what the model backbone expects)
_OUTPUT_CHANNELS = {
    ColorSpace.RGB: 3,
    ColorSpace.BGR: 3,
    ColorSpace.RGBA: 3,  # Alpha dropped
    ColorSpace.BGRA: 3,  # Alpha dropped
    ColorSpace.GREY: 1,
    ColorSpace.YUYV: 2,
    ColorSpace.NV12: 3,  # Converted to YUV444
    ColorSpace.NV21: 3,  # Converted to YUV444
    ColorSpace.RGGB: 1,
    ColorSpace.BGGR: 1,
    ColorSpace.GRBG: 1,
    ColorSpace.GBRG: 1,
}


def get_input_channels(color_space: ColorSpace | str) -> int:
    """Get the number of input channels for a color space.

    Args:
        color_space: ColorSpace enum value or string name.

    Returns:
        Number of input channels the model receives.

    Raises:
        ValueError: If the color space is not supported.
    """
    if isinstance(color_space, str):
        color_space = ColorSpace(color_space.lower())
    return _INPUT_CHANNELS[color_space]


def get_output_channels(color_space: ColorSpace | str) -> int:
    """Get the number of output channels for a color space.

    This represents what the model backbone will receive after the
    adaptor layer processes the input (e.g., RGBA -> 3 channels).

    Args:
        color_space: ColorSpace enum value or string name.

    Returns:
        Number of output channels after adaptor processing.

    Raises:
        ValueError: If the color space is not supported.
    """
    if isinstance(color_space, str):
        color_space = ColorSpace(color_space.lower())
    return _OUTPUT_CHANNELS[color_space]


# Phase 1 supported target formats (what the model will receive)
SUPPORTED_COLOR_SPACES: frozenset[ColorSpace] = frozenset(
    {
        ColorSpace.RGB,
        ColorSpace.BGR,
        ColorSpace.RGBA,
        ColorSpace.BGRA,
        ColorSpace.GREY,
        ColorSpace.YUYV,
    }
)

# Supported source formats (what the training data loader provides)
SUPPORTED_SOURCE_FORMATS: frozenset[ColorSpace] = frozenset(
    {
        ColorSpace.RGB,
        ColorSpace.RGBA,
        ColorSpace.BGR,
        ColorSpace.BGRA,
        ColorSpace.GREY,
    }
)


def is_supported(color_space: ColorSpace | str) -> bool:
    """Check if a color space is currently supported as a target format.

    Args:
        color_space: ColorSpace enum value or string name.

    Returns:
        True if the color space is supported as a target format.
    """
    if isinstance(color_space, str):
        try:
            color_space = ColorSpace(color_space.lower())
        except ValueError:
            return False
    return color_space in SUPPORTED_COLOR_SPACES


def is_supported_source(color_space: ColorSpace | str) -> bool:
    """Check if a color space is supported as a source format.

    Source formats are what the training data loader provides (e.g., RGB from
    most image libraries, BGR from OpenCV's default loading).

    Args:
        color_space: ColorSpace enum value or string name.

    Returns:
        True if the color space is supported as a source format.
    """
    if isinstance(color_space, str):
        try:
            color_space = ColorSpace(color_space.lower())
        except ValueError:
            return False
    return color_space in SUPPORTED_SOURCE_FORMATS


def get_source_channels(color_space: ColorSpace | str) -> int:
    """Get the number of channels for a source format.

    Args:
        color_space: ColorSpace enum value or string name.

    Returns:
        Number of channels in the source format.

    Raises:
        ValueError: If the color space is not a supported source format.
    """
    if isinstance(color_space, str):
        color_space = ColorSpace(color_space.lower())
    if color_space not in SUPPORTED_SOURCE_FORMATS:
        supported = ", ".join(cs.value for cs in SUPPORTED_SOURCE_FORMATS)
        raise ValueError(
            f"'{color_space.value}' is not a supported source format. "
            f"Supported: {supported}"
        )
    return _INPUT_CHANNELS[color_space]
