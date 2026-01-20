"""EdgeFirst Camera Adaptor - Train models for native camera formats.

This library helps train models that consume camera formats natively supported
by target edge platforms, avoiding costly runtime conversions.

Example:
    >>> from edgefirst.cameraadaptor import CameraAdaptorTransform, ColorSpace
    >>> from edgefirst.cameraadaptor.pytorch import CameraAdaptor
    >>>
    >>> # Create a transform for preprocessing (RGB source by default)
    >>> transform = CameraAdaptorTransform("yuyv")
    >>> yuyv_frame = transform(rgb_frame)
    >>>
    >>> # For OpenCV-loaded images (BGR by default)
    >>> transform = CameraAdaptorTransform("yuyv", source_format="bgr")
    >>> yuyv_frame = transform(bgr_frame)
    >>>
    >>> # Use the PyTorch module in your model
    >>> adaptor = CameraAdaptor("yuyv")
    >>>
    >>> # For channels-last input (e.g., from camera pipeline)
    >>> adaptor = CameraAdaptor("yuyv", channels_last=True)
"""

from ._version import __version__
from .color_spaces import (
    SUPPORTED_COLOR_SPACES,
    SUPPORTED_SOURCE_FORMATS,
    ColorSpace,
    get_input_channels,
    get_output_channels,
    get_source_channels,
    is_supported,
    is_supported_source,
)
from .config import CameraAdaptorConfig, create_config
from .hal_utils import (
    COLORSPACE_MAP,
    FOURCC_MAP,
    from_fourcc,
    from_fourcc_str,
    get_fourcc,
    get_fourcc_str,
)
from .transform import CameraAdaptorTransform

__all__ = [
    # Version
    "__version__",
    # Color spaces
    "ColorSpace",
    "SUPPORTED_COLOR_SPACES",
    "SUPPORTED_SOURCE_FORMATS",
    "get_input_channels",
    "get_output_channels",
    "get_source_channels",
    "is_supported",
    "is_supported_source",
    # Transform
    "CameraAdaptorTransform",
    # Config
    "CameraAdaptorConfig",
    "create_config",
    # HAL utilities
    "FOURCC_MAP",
    "COLORSPACE_MAP",
    "get_fourcc_str",
    "get_fourcc",
    "from_fourcc_str",
    "from_fourcc",
]
