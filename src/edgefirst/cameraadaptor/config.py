"""Configuration and metadata support for camera adaptor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .color_spaces import (
    SUPPORTED_COLOR_SPACES,
    ColorSpace,
    get_input_channels,
    get_output_channels,
)

DType = Literal["float32", "float16", "uint8", "int8"]


@dataclass
class CameraAdaptorConfig:
    """Configuration for camera adaptor.

    This dataclass holds the configuration for a camera adaptor, including
    the target color space and optional data type specifications for
    quantized models.

    Args:
        adaptor: Target color space name. Defaults to "rgb".
        input_dtype: Input data type for quantized models.
            Defaults to "float32".
        output_dtype: Output data type for quantized models.
            Defaults to "float32".

    Example:
        >>> config = CameraAdaptorConfig(adaptor="yuyv", input_dtype="uint8")
        >>> print(config.input_channels)  # 2
        >>> print(config.to_metadata())
        {'camera_adaptor': 'yuyv', 'input_channels': 2, ...}
    """

    adaptor: str = "rgb"
    input_dtype: DType = "float32"
    output_dtype: DType = "float32"

    # Computed fields (set in __post_init__)
    _color_space: ColorSpace | None = field(
        default=None, repr=False, init=False
    )

    def __post_init__(self) -> None:
        """Validate and set computed fields after initialization."""
        adaptor_lower = self.adaptor.lower()
        try:
            self._color_space = ColorSpace(adaptor_lower)
        except ValueError:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Invalid camera adaptor: {self.adaptor}. "
                f"Supported formats: {supported}"
            ) from None

        if self._color_space not in SUPPORTED_COLOR_SPACES:
            supported = ", ".join(cs.value for cs in SUPPORTED_COLOR_SPACES)
            raise ValueError(
                f"Color space '{self._color_space.value}' not yet supported. "
                f"Currently supported: {supported}"
            )

        # Normalize adaptor name
        self.adaptor = adaptor_lower

        # Validate dtypes
        valid_dtypes = {"float32", "float16", "uint8", "int8"}
        if self.input_dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid input_dtype: {self.input_dtype}. "
                f"Valid options: {valid_dtypes}"
            )
        if self.output_dtype not in valid_dtypes:
            raise ValueError(
                f"Invalid output_dtype: {self.output_dtype}. "
                f"Valid options: {valid_dtypes}"
            )

    @property
    def color_space(self) -> ColorSpace:
        """Get the color space enum value."""
        assert self._color_space is not None
        return self._color_space

    @property
    def input_channels(self) -> int:
        """Get the number of input channels."""
        assert self._color_space is not None
        return get_input_channels(self._color_space)

    @property
    def output_channels(self) -> int:
        """Get the number of output channels after adaptor processing."""
        assert self._color_space is not None
        return get_output_channels(self._color_space)

    @property
    def is_quantized(self) -> bool:
        """Check if this config specifies quantized data types."""
        return self.input_dtype in ("uint8", "int8") or self.output_dtype in (
            "uint8",
            "int8",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary.

        Returns:
            Dictionary representation of the config.
        """
        return {
            "adaptor": self.adaptor,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
        }

    def to_metadata(self) -> dict[str, Any]:
        """Convert config to metadata dictionary for model export.

        This format is suitable for embedding in exported model metadata.

        Returns:
            Metadata dictionary.
        """
        return {
            "camera_adaptor": self.adaptor,
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "input_dtype": self.input_dtype,
            "output_dtype": self.output_dtype,
            "color_space": self._color_space.value
            if self._color_space
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraAdaptorConfig:
        """Create config from a dictionary.

        Args:
            data: Dictionary with config values.

        Returns:
            New CameraAdaptorConfig instance.
        """
        return cls(
            adaptor=data.get("adaptor", "rgb"),
            input_dtype=data.get("input_dtype", "float32"),
            output_dtype=data.get("output_dtype", "float32"),
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> CameraAdaptorConfig:
        """Create config from model metadata.

        Args:
            metadata: Metadata dictionary from exported model.

        Returns:
            New CameraAdaptorConfig instance.
        """
        adaptor = metadata.get("camera_adaptor") or metadata.get(
            "adaptor", "rgb"
        )
        return cls(
            adaptor=adaptor,
            input_dtype=metadata.get("input_dtype", "float32"),
            output_dtype=metadata.get("output_dtype", "float32"),
        )


def create_config(
    adaptor: str = "rgb",
    input_dtype: DType = "float32",
    output_dtype: DType = "float32",
) -> CameraAdaptorConfig:
    """Create a camera adaptor configuration.

    This is a convenience function for creating CameraAdaptorConfig instances.

    Args:
        adaptor: Target color space name. Defaults to "rgb".
        input_dtype: Input data type. Defaults to "float32".
        output_dtype: Output data type. Defaults to "float32".

    Returns:
        New CameraAdaptorConfig instance.
    """
    return CameraAdaptorConfig(
        adaptor=adaptor,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


__all__ = ["CameraAdaptorConfig", "create_config", "DType"]
