"""Tests for config module."""

import pytest

from edgefirst.cameraadaptor import ColorSpace
from edgefirst.cameraadaptor.config import (
    CameraAdaptorConfig,
    create_config,
)


class TestCameraAdaptorConfig:
    """Tests for CameraAdaptorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = CameraAdaptorConfig()
        assert config.adaptor == "rgb"
        assert config.input_dtype == "float32"
        assert config.output_dtype == "float32"

    def test_custom_adaptor(self):
        """Test custom adaptor configuration."""
        config = CameraAdaptorConfig(adaptor="yuyv")
        assert config.adaptor == "yuyv"
        assert config.color_space == ColorSpace.YUYV

    def test_case_normalization(self):
        """Test adaptor name is normalized to lowercase."""
        config = CameraAdaptorConfig(adaptor="YUYV")
        assert config.adaptor == "yuyv"

    def test_invalid_adaptor(self):
        """Test invalid adaptor raises ValueError."""
        with pytest.raises(ValueError, match="Invalid camera adaptor"):
            CameraAdaptorConfig(adaptor="invalid")

    def test_unsupported_adaptor(self):
        """Test unsupported adaptor raises ValueError."""
        with pytest.raises(ValueError, match="not yet supported"):
            CameraAdaptorConfig(adaptor="nv12")

    def test_invalid_input_dtype(self):
        """Test invalid input_dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid input_dtype"):
            CameraAdaptorConfig(input_dtype="invalid")

    def test_invalid_output_dtype(self):
        """Test invalid output_dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid output_dtype"):
            CameraAdaptorConfig(output_dtype="invalid")

    def test_input_channels(self):
        """Test input_channels property."""
        assert CameraAdaptorConfig(adaptor="rgb").input_channels == 3
        assert CameraAdaptorConfig(adaptor="rgba").input_channels == 4
        assert CameraAdaptorConfig(adaptor="yuyv").input_channels == 2

    def test_output_channels(self):
        """Test output_channels property."""
        assert CameraAdaptorConfig(adaptor="rgb").output_channels == 3
        assert CameraAdaptorConfig(adaptor="rgba").output_channels == 3
        assert CameraAdaptorConfig(adaptor="yuyv").output_channels == 2

    def test_is_quantized_float(self):
        """Test is_quantized for float dtypes."""
        config = CameraAdaptorConfig(
            input_dtype="float32", output_dtype="float32"
        )
        assert not config.is_quantized

    def test_is_quantized_uint8(self):
        """Test is_quantized for uint8 dtype."""
        config = CameraAdaptorConfig(input_dtype="uint8")
        assert config.is_quantized

    def test_is_quantized_int8(self):
        """Test is_quantized for int8 dtype."""
        config = CameraAdaptorConfig(output_dtype="int8")
        assert config.is_quantized


class TestCameraAdaptorConfigSerialization:
    """Tests for config serialization."""

    def test_to_dict(self):
        """Test to_dict method."""
        config = CameraAdaptorConfig(adaptor="rgba", input_dtype="uint8")
        d = config.to_dict()
        assert d["adaptor"] == "rgba"
        assert d["input_dtype"] == "uint8"
        assert d["input_channels"] == 4
        assert d["output_channels"] == 3

    def test_to_metadata(self):
        """Test to_metadata method."""
        config = CameraAdaptorConfig(adaptor="yuyv")
        metadata = config.to_metadata()
        assert metadata["camera_adaptor"] == "yuyv"
        assert metadata["color_space"] == "yuyv"
        assert metadata["input_channels"] == 2

    def test_from_dict(self):
        """Test from_dict class method."""
        d = {"adaptor": "bgra", "input_dtype": "uint8", "output_dtype": "int8"}
        config = CameraAdaptorConfig.from_dict(d)
        assert config.adaptor == "bgra"
        assert config.input_dtype == "uint8"
        assert config.output_dtype == "int8"

    def test_from_dict_defaults(self):
        """Test from_dict with missing values uses defaults."""
        d = {"adaptor": "yuyv"}
        config = CameraAdaptorConfig.from_dict(d)
        assert config.adaptor == "yuyv"
        assert config.input_dtype == "float32"
        assert config.output_dtype == "float32"

    def test_from_metadata(self):
        """Test from_metadata class method."""
        metadata = {
            "camera_adaptor": "rgba",
            "input_dtype": "uint8",
            "output_dtype": "float32",
        }
        config = CameraAdaptorConfig.from_metadata(metadata)
        assert config.adaptor == "rgba"
        assert config.input_dtype == "uint8"

    def test_from_metadata_legacy_key(self):
        """Test from_metadata with legacy 'adaptor' key."""
        metadata = {"adaptor": "bgr"}
        config = CameraAdaptorConfig.from_metadata(metadata)
        assert config.adaptor == "bgr"

    def test_roundtrip(self):
        """Test config roundtrip through dict."""
        original = CameraAdaptorConfig(adaptor="rgba", input_dtype="uint8")
        d = original.to_dict()
        restored = CameraAdaptorConfig.from_dict(d)
        assert original.adaptor == restored.adaptor
        assert original.input_dtype == restored.input_dtype
        assert original.output_dtype == restored.output_dtype

    def test_metadata_roundtrip_all_fields(self):
        """Test config roundtrip through metadata preserves all fields.

        This ensures that all CameraAdaptorConfig fields survive serialization
        to metadata and deserialization back, which is critical for model
        deployment workflows.
        """
        # Test with all non-default values to ensure proper roundtrip
        original = CameraAdaptorConfig(
            adaptor="rgba",
            input_dtype="uint8",
            output_dtype="int8",
        )

        # Roundtrip through metadata
        metadata = original.to_metadata()
        restored = CameraAdaptorConfig.from_metadata(metadata)

        # Verify all fields match
        assert original.adaptor == restored.adaptor, "adaptor mismatch"
        assert original.input_dtype == restored.input_dtype, (
            "input_dtype mismatch"
        )
        assert original.output_dtype == restored.output_dtype, (
            "output_dtype mismatch"
        )

        # Verify derived properties also match
        assert original.color_space == restored.color_space, (
            "color_space mismatch"
        )
        assert original.input_channels == restored.input_channels, (
            "input_channels mismatch"
        )
        assert original.output_channels == restored.output_channels, (
            "output_channels mismatch"
        )
        assert original.is_quantized == restored.is_quantized, (
            "is_quantized mismatch"
        )

    def test_metadata_roundtrip_all_adaptors(self):
        """Test metadata roundtrip for all supported adaptors."""
        from edgefirst.cameraadaptor import SUPPORTED_COLOR_SPACES

        for color_space in SUPPORTED_COLOR_SPACES:
            original = CameraAdaptorConfig(adaptor=color_space.value)
            metadata = original.to_metadata()
            restored = CameraAdaptorConfig.from_metadata(metadata)

            assert original.adaptor == restored.adaptor, (
                f"{color_space.value}: adaptor mismatch"
            )
            assert original.input_channels == restored.input_channels, (
                f"{color_space.value}: channels mismatch"
            )


class TestCreateConfig:
    """Tests for create_config factory function."""

    def test_create_default(self):
        """Test create_config with defaults."""
        config = create_config()
        assert config.adaptor == "rgb"

    def test_create_custom(self):
        """Test create_config with custom values."""
        config = create_config(adaptor="yuyv", input_dtype="uint8")
        assert config.adaptor == "yuyv"
        assert config.input_dtype == "uint8"
