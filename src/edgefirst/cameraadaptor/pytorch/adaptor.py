"""PyTorch CameraAdaptor module."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..color_spaces import (
    SUPPORTED_COLOR_SPACES,
    ColorSpace,
    get_input_channels,
    get_output_channels,
)


class CameraAdaptor(nn.Module):
    """Camera adaptor module for PyTorch models.

    This module adapts camera input formats for neural network processing.
    It is designed to be placed at the beginning of a model to handle
    different input color spaces and channel orderings.

    For RGBA/BGRA inputs, the module drops the alpha channel to produce
    3-channel output. For other formats, it passes through unchanged.

    Important:
        This layer does NOT perform color space conversion. Color conversion
        is handled by ``CameraAdaptorTransform`` during training data loading,
        which converts RGB training images to the target camera format (e.g.,
        YUYV, BGR). At inference time, the camera or ISP provides data directly
        in the target format, so no conversion is needed in the model itself.

        This layer only performs:
        1. Layout permutation (NHWC ↔ NCHW) when ``channels_last=True``
        2. Alpha channel dropping for RGBA/BGRA inputs

    Args:
        adaptor: Target color space as string or ColorSpace enum.
            Defaults to "rgb".
        channels_last: If True, input is expected in NHWC format
            (channels-last) and will be permuted to NCHW for PyTorch.
            Useful for camera data in channels-last format.
            Defaults to False.
        validate_channels: If True, validates input channel count.
            Raises ValueError on mismatch. Defaults to True.
            Set to False for performance-critical inference.

    Example:
        >>> # Standard PyTorch NCHW input
        >>> adaptor = CameraAdaptor("rgba")
        >>> x = torch.randn(1, 4, 224, 224)  # NCHW
        >>> y = adaptor(x)  # Returns (1, 3, 224, 224)
        >>>
        >>> # Channels-last input (e.g., from camera pipeline)
        >>> adaptor = CameraAdaptor("yuyv", channels_last=True)
        >>> x = torch.randn(1, 224, 224, 2)  # NHWC
        >>> y = adaptor(x)  # Returns (1, 2, 224, 224) in NCHW

    YAML Configuration (for ultralytics):
        ```yaml
        backbone:
          - [-1, 1, CameraAdaptor, [rgba]]
        ```

    Note:
        When channels_last=True, the exported model will contain a permute
        operation at the beginning, allowing the model to accept NHWC input
        directly from camera pipelines while internally using PyTorch's
        native NCHW format.
    """

    def __init__(
        self,
        adaptor: str | ColorSpace = "rgb",
        channels_last: bool = False,
        validate_channels: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(adaptor, ColorSpace):
            self._color_space = adaptor
            self._adaptor_name = adaptor.value
        else:
            adaptor_lower = adaptor.lower()
            try:
                self._color_space = ColorSpace(adaptor_lower)
            except ValueError:
                supported = ", ".join(
                    cs.value for cs in SUPPORTED_COLOR_SPACES
                )
                raise ValueError(
                    f"Invalid camera adaptor: {adaptor}. "
                    f"Supported formats: {supported}"
                ) from None
            self._adaptor_name = adaptor_lower

        if self._color_space not in SUPPORTED_COLOR_SPACES:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Color space '{self._color_space.value}' not yet supported. "
                f"Currently supported: {supported}"
            )

        self._channels_last = channels_last
        self._validate_channels = validate_channels

        # Register adaptor name as buffer for model serialization
        self.register_buffer(
            "_adaptor_tensor",
            torch.zeros(1),  # Placeholder, actual name stored in _adaptor_name
            persistent=False,
        )

    @property
    def adaptor(self) -> str:
        """Get the adaptor name as a string."""
        return self._adaptor_name

    @property
    def color_space(self) -> ColorSpace:
        """Get the color space enum value."""
        return self._color_space

    @property
    def channels_last(self) -> bool:
        """Get whether input is expected in channels-last format."""
        return self._channels_last

    @property
    def validate_channels(self) -> bool:
        """Get whether input channel validation is enabled."""
        return self._validate_channels

    @staticmethod
    def compute_input_channels(args: str | list) -> int:
        """Compute the number of input channels for YAML configuration.

        This static method is called by ultralytics model parser to
        determine input channels from YAML args.

        Args:
            args: Adaptor name as string, or list where first element is
                adaptor name and optional second element is channels_last bool.

        Returns:
            Number of input channels for the specified adaptor.
        """
        if isinstance(args, list):  # noqa: SIM108
            adaptor = args[0]
        else:
            adaptor = args
        if isinstance(adaptor, str):
            adaptor = adaptor.lower()
        return get_input_channels(adaptor)

    @staticmethod
    def compute_output_channels(args: str | list) -> int:
        """Compute the number of output channels for YAML configuration.

        This static method is called by ultralytics model parser to
        determine output channels from YAML args.

        Args:
            args: Adaptor name as string, or list where first element is
                adaptor name and optional second element is channels_last bool.

        Returns:
            Number of output channels for the specified adaptor.
        """
        if isinstance(args, list):  # noqa: SIM108
            adaptor = args[0]
        else:
            adaptor = args
        if isinstance(adaptor, str):
            adaptor = adaptor.lower()
        return get_output_channels(adaptor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the camera adaptor.

        Processing order:
        1. Validate input channels (if validate_channels=True)
        2. If channels_last=True: permute NHWC → NCHW
        3. If RGBA/BGRA: drop alpha channel (4 → 3 channels)

        Args:
            x: Input tensor with shape (N, C, H, W) if channels_last=False,
               or (N, H, W, C) if channels_last=True.

        Returns:
            Output tensor in NCHW format, potentially with reduced channels.

        Raises:
            ValueError: If validate_channels=True and input has wrong
                channel count.
        """
        # Step 1: Validate input channels
        if self._validate_channels:
            expected = get_input_channels(self._color_space)
            actual = x.shape[-1] if self._channels_last else x.shape[1]
            if actual != expected:
                layout = "NHWC" if self._channels_last else "NCHW"
                raise ValueError(
                    f"CameraAdaptor({self._adaptor_name}): "
                    f"expected {expected} channels, got {actual}. "
                    f"Shape: {tuple(x.shape)} ({layout})"
                )

        # Step 2: Handle channel ordering
        if self._channels_last:
            # NHWC -> NCHW: (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2)

        # Step 3: Handle alpha channel removal
        if self._color_space in (ColorSpace.RGBA, ColorSpace.BGRA):
            return x[:, :3, :, :]

        return x

    def extra_repr(self) -> str:
        """Return extra representation string for printing."""
        parts = [f"adaptor='{self._adaptor_name}'"]
        if self._channels_last:
            parts.append("channels_last=True")
        if not self._validate_channels:
            parts.append("validate_channels=False")
        return ", ".join(parts)
