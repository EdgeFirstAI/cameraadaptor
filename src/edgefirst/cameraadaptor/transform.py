"""Camera adaptor transform for preprocessing images."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from .color_spaces import (
    SUPPORTED_COLOR_SPACES,
    SUPPORTED_SOURCE_FORMATS,
    ColorSpace,
    get_input_channels,
    get_output_channels,
    get_source_channels,
)

if TYPE_CHECKING:
    pass


class CameraAdaptorTransform:
    """Transform for converting source images to target camera formats.

    This transform is used during training to convert images from your data
    loader's format into the target camera format that will be used at
    inference time on the edge device.

    Args:
        adaptor: Target color space as string or ColorSpace enum.
            Defaults to "rgb".
        source_format: Source color space from your data loader.
            Defaults to "rgb". Use "bgr" for OpenCV's cv2.imread.

    Raises:
        ValueError: If the adaptor or source format is not supported.
        ImportError: If OpenCV is not installed (required for conversions).

    Example:
        >>> # Standard RGB dataset (PIL, torchvision, etc.)
        >>> transform = CameraAdaptorTransform("yuyv")
        >>> yuyv_frame = transform(rgb_frame)
        >>>
        >>> # OpenCV-loaded images (BGR by default)
        >>> transform = CameraAdaptorTransform("yuyv", source_format="bgr")
        >>> yuyv_frame = transform(bgr_frame)

    Note:
        OpenCV loads images as BGR by default. If you're using cv2.imread()
        without explicit color conversion, set source_format="bgr".

        Common data loader formats:
        - PIL/Pillow: RGB
        - torchvision: RGB
        - OpenCV cv2.imread(): BGR (default)
        - OpenCV cv2.imread() with IMREAD_UNCHANGED: BGRA (PNG alpha)
        - imageio: RGB
        - skimage: RGB
    """

    def __init__(
        self,
        adaptor: str | ColorSpace = "rgb",
        source_format: str | ColorSpace = "rgb",
    ) -> None:
        # Parse and validate target format
        if isinstance(adaptor, str):
            adaptor_lower = adaptor.lower()
            try:
                self._color_space = ColorSpace(adaptor_lower)
            except ValueError:
                supported = ", ".join(
                    cs.value for cs in SUPPORTED_COLOR_SPACES
                )
                raise ValueError(
                    f"Invalid target adaptor: {adaptor}. "
                    f"Supported formats: {supported}"
                ) from None
        else:
            self._color_space = adaptor

        if self._color_space not in SUPPORTED_COLOR_SPACES:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Target '{self._color_space.value}' not yet supported. "
                f"Currently supported: {supported}"
            )

        # Parse and validate source format
        if isinstance(source_format, str):
            source_lower = source_format.lower()
            try:
                self._source_format = ColorSpace(source_lower)
            except ValueError:
                supported = ", ".join(
                    cs.value for cs in SUPPORTED_SOURCE_FORMATS
                )
                raise ValueError(
                    f"Invalid source format: {source_format}. "
                    f"Supported formats: {supported}"
                ) from None
        else:
            self._source_format = source_format

        if self._source_format not in SUPPORTED_SOURCE_FORMATS:
            supported = ", ".join(cs.value for cs in SUPPORTED_SOURCE_FORMATS)
            raise ValueError(
                f"Source '{self._source_format.value}' not supported. "
                f"Supported: {supported}"
            )

        self._adaptor_name = self._color_space.value
        self._source_name = self._source_format.value

    @property
    def adaptor(self) -> str:
        """Get the target adaptor name as a string."""
        return self._adaptor_name

    @property
    def color_space(self) -> ColorSpace:
        """Get the target color space enum value."""
        return self._color_space

    @property
    def source_format(self) -> str:
        """Get the source format name as a string."""
        return self._source_name

    @property
    def source_color_space(self) -> ColorSpace:
        """Get the source color space enum value."""
        return self._source_format

    @property
    def channels(self) -> int:
        """Get the number of output channels for this transform."""
        return get_input_channels(self._color_space)

    @property
    def input_channels(self) -> int:
        """Get the number of input channels from the source format."""
        return get_source_channels(self._source_format)

    @property
    def output_channels(self) -> int:
        """Get the number of channels the model backbone receives."""
        return get_output_channels(self._color_space)

    def _get_converter(self) -> Callable[[np.ndarray], np.ndarray]:
        """Get the conversion function for source → target color space."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required for color space conversions. "
                "Install with: pip install edgefirst-cameraadaptor[transform]"
            ) from None

        source = self._source_format
        target = self._color_space

        # Identity case
        if source == target:
            return lambda frame: frame

        # Build conversion lookup table
        # Key: (source, target) -> OpenCV conversion code or callable
        conversions: dict[tuple[ColorSpace, ColorSpace], int | Callable] = {
            # From RGB
            (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
            (ColorSpace.RGB, ColorSpace.RGBA): cv2.COLOR_RGB2RGBA,
            (ColorSpace.RGB, ColorSpace.BGRA): cv2.COLOR_RGB2BGRA,
            (ColorSpace.RGB, ColorSpace.YUYV): cv2.COLOR_RGB2YUV_YUYV,
            (ColorSpace.RGB, ColorSpace.GREY): cv2.COLOR_RGB2GRAY,
            # From BGR
            (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
            (ColorSpace.BGR, ColorSpace.RGBA): cv2.COLOR_BGR2RGBA,
            (ColorSpace.BGR, ColorSpace.BGRA): cv2.COLOR_BGR2BGRA,
            (ColorSpace.BGR, ColorSpace.YUYV): cv2.COLOR_BGR2YUV_YUYV,
            (ColorSpace.BGR, ColorSpace.GREY): cv2.COLOR_BGR2GRAY,
            # From RGBA
            (ColorSpace.RGBA, ColorSpace.RGB): cv2.COLOR_RGBA2RGB,
            (ColorSpace.RGBA, ColorSpace.BGR): cv2.COLOR_RGBA2BGR,
            (ColorSpace.RGBA, ColorSpace.BGRA): cv2.COLOR_RGBA2BGRA,
            (ColorSpace.RGBA, ColorSpace.GREY): cv2.COLOR_RGBA2GRAY,
            # From BGRA
            (ColorSpace.BGRA, ColorSpace.RGB): cv2.COLOR_BGRA2RGB,
            (ColorSpace.BGRA, ColorSpace.BGR): cv2.COLOR_BGRA2BGR,
            (ColorSpace.BGRA, ColorSpace.RGBA): cv2.COLOR_BGRA2RGBA,
            (ColorSpace.BGRA, ColorSpace.GREY): cv2.COLOR_BGRA2GRAY,
        }

        key = (source, target)

        if key in conversions:
            code = conversions[key]
            if callable(code):
                return code
            # Create a converter function with the OpenCV code
            cv_code: int = code

            def make_converter(c: int) -> Callable[[np.ndarray], np.ndarray]:
                return lambda frame: cv2.cvtColor(frame, c)

            return make_converter(cv_code)

        # Handle multi-step conversions (e.g., RGBA → YUYV)
        # First convert to RGB/BGR, then to target
        if (
            source in (ColorSpace.RGBA, ColorSpace.BGRA)
            and target == ColorSpace.YUYV
        ):
            # Strip alpha first, then convert to YUYV
            if source == ColorSpace.RGBA:
                return lambda frame: cv2.cvtColor(
                    cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB),
                    cv2.COLOR_RGB2YUV_YUYV,
                )
            else:  # BGRA
                return lambda frame: cv2.cvtColor(
                    cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR),
                    cv2.COLOR_BGR2YUV_YUYV,
                )

        # Should not reach here if validation is correct
        raise ValueError(
            f"Unsupported conversion: {source.value} → {target.value}"
        )

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Convert a source frame to the target color space.

        Args:
            frame: Input image as numpy array with shape (H, W, C) where C
                matches the source format's channel count.

        Returns:
            Converted image in the target color space.
        """
        converter = self._get_converter()
        return converter(frame)

    def convert(self, frame: np.ndarray) -> np.ndarray:
        """Convert a source frame to the target color space.

        This is an alias for __call__ for compatibility with modelpack.

        Args:
            frame: Input image as numpy array with shape (H, W, C).

        Returns:
            Converted image in the target color space.
        """
        return self(frame)

    def __str__(self) -> str:
        """Return the adaptor name."""
        return self._adaptor_name

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        if self._source_name == "rgb":
            return f"CameraAdaptorTransform(adaptor='{self._adaptor_name}')"
        return (
            f"CameraAdaptorTransform(adaptor='{self._adaptor_name}', "
            f"source_format='{self._source_name}')"
        )

    def get_num_channels(self) -> int:
        """Get the number of channels in the output frame.

        This is an alias for the channels property for backward compatibility.
        """
        return self.channels
