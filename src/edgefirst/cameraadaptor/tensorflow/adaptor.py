"""TensorFlow/Keras CameraAdaptor layer."""

from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf
except ImportError as err:
    raise ImportError(
        "TensorFlow is required for this module. "
        "Install with: pip install edgefirst-cameraadaptor[tensorflow]"
    ) from err

from ..color_spaces import (
    SUPPORTED_COLOR_SPACES,
    ColorSpace,
    get_input_channels,
)


@tf.keras.utils.register_keras_serializable(
    package="edgefirst.cameraadaptor", name="CameraAdaptor"
)
class CameraAdaptor(tf.keras.layers.Layer):
    """Camera adaptor layer for Keras models.

    This layer adapts camera input formats for neural network processing.
    It can be placed at the beginning of a model to handle different
    input color spaces and channel orderings.

    For RGBA/BGRA inputs (4 channels), the layer drops the alpha channel
    to produce 3-channel output. Other formats pass through unchanged.

    Important:
        This layer does NOT perform color space conversion. Color conversion
        is handled by ``CameraAdaptorTransform`` during training data loading,
        which converts RGB training images to the target camera format (e.g.,
        YUYV, BGR). At inference time, the camera or ISP provides data directly
        in the target format, so no conversion is needed in the model itself.

        This layer only performs:
        1. Layout permutation (NCHW ↔ NHWC) when ``channels_first=True``
        2. Alpha channel dropping for RGBA/BGRA inputs

    Args:
        adaptor: Target color space as string. Required. Supported:
            rgb, bgr, rgba, bgra, grey, yuyv.
        channels_first: If True, input is expected in NCHW format
            (channels-first) and will be permuted to NHWC for TensorFlow.
            Useful for PyTorch-style inputs. Defaults to False.
        validate_channels: If True, validates input channel count.
            Raises ValueError on mismatch. Defaults to True.
            Set to False for performance-critical inference.
        **kwargs: Additional layer arguments.

    Example:
        >>> # Standard TensorFlow NHWC input
        >>> layer = CameraAdaptor("rgba")
        >>> x = tf.random.normal((1, 224, 224, 4))  # NHWC
        >>> y = layer(x)  # Returns (1, 224, 224, 3)
        >>>
        >>> # Channels-first input (PyTorch-style)
        >>> layer = CameraAdaptor("yuyv", channels_first=True)
        >>> x = tf.random.normal((1, 2, 224, 224))  # NCHW
        >>> y = layer(x)  # Returns (1, 224, 224, 2) in NHWC

    Note:
        TensorFlow uses NHWC (channels-last) by default. The channels_first
        option is provided for compatibility with pipelines that produce
        NCHW data, converting it to TensorFlow's native format.
    """

    def __init__(
        self,
        adaptor: str,
        channels_first: bool = False,
        validate_channels: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._adaptor_name: str = ""
        self._color_space: ColorSpace = (
            ColorSpace.RGB
        )  # Will be set by _set_adaptor
        self._channels_first = channels_first
        self._validate_channels = validate_channels

        self._set_adaptor(adaptor)

    def _set_adaptor(self, adaptor: str) -> None:
        """Set the adaptor configuration.

        Args:
            adaptor: Adaptor name as string.

        Raises:
            ValueError: If the adaptor is not supported.
        """
        adaptor_lower = adaptor.lower()
        try:
            self._color_space = ColorSpace(adaptor_lower)
        except ValueError:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Invalid camera adaptor: {adaptor}. "
                f"Supported formats: {supported}"
            ) from None

        if self._color_space not in SUPPORTED_COLOR_SPACES:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Color space '{self._color_space.value}' not yet supported. "
                f"Currently supported: {supported}"
            )

        self._adaptor_name = adaptor_lower

    @property
    def adaptor(self) -> str:
        """Get the adaptor name as a string."""
        return self._adaptor_name

    @property
    def color_space(self) -> ColorSpace:
        """Get the color space enum value."""
        return self._color_space

    @property
    def channels_first(self) -> bool:
        """Get whether input is expected in channels-first format."""
        return self._channels_first

    @property
    def validate_channels(self) -> bool:
        """Get whether input channel validation is enabled."""
        return self._validate_channels

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape of input tensor (N, H, W, C) or (N, C, H, W)
                depending on channels_first setting.
        """
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Process inputs through the camera adaptor.

        Processing order:
        1. Validate input channels (if validate_channels=True)
        2. If channels_first=True: permute NCHW → NHWC
        3. If RGBA/BGRA: drop alpha channel (4 → 3 channels)

        Args:
            inputs: Input tensor with shape (N, H, W, C) if
                channels_first=False, or (N, C, H, W) if channels_first=True.

        Returns:
            Output tensor in NHWC format, potentially with reduced channels.

        Raises:
            ValueError: If validate_channels=True and input has wrong
                channel count.
        """
        x = inputs

        # Step 1: Validate input channels
        if self._validate_channels:
            expected = get_input_channels(self._color_space)
            input_shape = tf.shape(x)
            actual = (
                input_shape[1] if self._channels_first else input_shape[-1]
            )
            # Use tf.debugging.assert for graph-compatible validation
            tf.debugging.assert_equal(
                actual,
                expected,
                message=(
                    f"CameraAdaptor({self._adaptor_name}): "
                    f"expected {expected} channels"
                ),
            )

        # Step 2: Handle channel ordering
        if self._channels_first:
            # NCHW -> NHWC: (N, C, H, W) -> (N, H, W, C)
            x = tf.transpose(x, perm=[0, 2, 3, 1])

        # Step 3: Handle alpha channel removal
        if self._color_space in (ColorSpace.RGBA, ColorSpace.BGRA):
            return x[:, :, :, :3]

        return x

    def compute_output_shape(
        self, input_shape: tf.TensorShape
    ) -> tf.TensorShape:
        """Compute output shape based on input shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Shape of output tensor (always in NHWC format).
        """
        if self._channels_first:
            # Input is NCHW, output will be NHWC
            n, c, h, w = input_shape
            if self._color_space in (ColorSpace.RGBA, ColorSpace.BGRA):
                return tf.TensorShape([n, h, w, 3])
            return tf.TensorShape([n, h, w, c])
        else:
            # Input and output are both NHWC
            if self._color_space in (ColorSpace.RGBA, ColorSpace.BGRA):
                return tf.TensorShape(
                    [input_shape[0], input_shape[1], input_shape[2], 3]
                )
            return input_shape

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config: dict[str, Any] = super().get_config()
        config["adaptor"] = self._adaptor_name
        config["channels_first"] = self._channels_first
        config["validate_channels"] = self._validate_channels
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CameraAdaptor:
        """Create layer from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            New CameraAdaptor instance.
        """
        return cls(**config)


__all__ = ["CameraAdaptor"]
