"""Tests for color_spaces module."""

import pytest

from edgefirst.cameraadaptor.color_spaces import (
    SUPPORTED_COLOR_SPACES,
    SUPPORTED_SOURCE_FORMATS,
    ColorSpace,
    get_input_channels,
    get_output_channels,
    get_source_channels,
    is_supported,
    is_supported_source,
)


class TestColorSpace:
    """Tests for ColorSpace enum."""

    def test_rgb_value(self):
        """Test RGB color space value."""
        assert ColorSpace.RGB.value == "rgb"

    def test_bgr_value(self):
        """Test BGR color space value."""
        assert ColorSpace.BGR.value == "bgr"

    def test_rgba_value(self):
        """Test RGBA color space value."""
        assert ColorSpace.RGBA.value == "rgba"

    def test_bgra_value(self):
        """Test BGRA color space value."""
        assert ColorSpace.BGRA.value == "bgra"

    def test_yuyv_value(self):
        """Test YUYV color space value."""
        assert ColorSpace.YUYV.value == "yuyv"

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert ColorSpace("rgb") == ColorSpace.RGB
        assert ColorSpace("yuyv") == ColorSpace.YUYV

    def test_invalid_color_space(self):
        """Test that invalid color space raises ValueError."""
        with pytest.raises(ValueError):
            ColorSpace("invalid")


class TestGetInputChannels:
    """Tests for get_input_channels function."""

    def test_rgb_channels(self):
        """Test RGB input channels."""
        assert get_input_channels(ColorSpace.RGB) == 3
        assert get_input_channels("rgb") == 3

    def test_bgr_channels(self):
        """Test BGR input channels."""
        assert get_input_channels(ColorSpace.BGR) == 3
        assert get_input_channels("bgr") == 3

    def test_rgba_channels(self):
        """Test RGBA input channels."""
        assert get_input_channels(ColorSpace.RGBA) == 4
        assert get_input_channels("rgba") == 4

    def test_bgra_channels(self):
        """Test BGRA input channels."""
        assert get_input_channels(ColorSpace.BGRA) == 4
        assert get_input_channels("bgra") == 4

    def test_yuyv_channels(self):
        """Test YUYV input channels."""
        assert get_input_channels(ColorSpace.YUYV) == 2
        assert get_input_channels("yuyv") == 2

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert get_input_channels("RGB") == 3
        assert get_input_channels("YUYV") == 2


class TestGetOutputChannels:
    """Tests for get_output_channels function."""

    def test_rgb_output(self):
        """Test RGB output channels."""
        assert get_output_channels(ColorSpace.RGB) == 3

    def test_rgba_output_drops_alpha(self):
        """Test RGBA output drops alpha channel."""
        assert get_output_channels(ColorSpace.RGBA) == 3

    def test_bgra_output_drops_alpha(self):
        """Test BGRA output drops alpha channel."""
        assert get_output_channels(ColorSpace.BGRA) == 3

    def test_yuyv_output(self):
        """Test YUYV output channels."""
        assert get_output_channels(ColorSpace.YUYV) == 2


class TestSupportedColorSpaces:
    """Tests for SUPPORTED_COLOR_SPACES."""

    def test_phase1_formats(self):
        """Test that Phase 1 formats are supported."""
        assert ColorSpace.RGB in SUPPORTED_COLOR_SPACES
        assert ColorSpace.BGR in SUPPORTED_COLOR_SPACES
        assert ColorSpace.RGBA in SUPPORTED_COLOR_SPACES
        assert ColorSpace.BGRA in SUPPORTED_COLOR_SPACES
        assert ColorSpace.YUYV in SUPPORTED_COLOR_SPACES

    def test_is_frozenset(self):
        """Test that SUPPORTED_COLOR_SPACES is immutable."""
        assert isinstance(SUPPORTED_COLOR_SPACES, frozenset)


class TestIsSupported:
    """Tests for is_supported function."""

    def test_supported_formats(self):
        """Test supported formats return True."""
        assert is_supported(ColorSpace.RGB)
        assert is_supported("rgb")
        assert is_supported("yuyv")

    def test_unsupported_formats(self):
        """Test unsupported formats return False."""
        assert not is_supported(ColorSpace.NV12)
        assert not is_supported("nv12")

    def test_invalid_format(self):
        """Test invalid format returns False."""
        assert not is_supported("invalid")


class TestSupportedSourceFormats:
    """Tests for SUPPORTED_SOURCE_FORMATS."""

    def test_source_formats(self):
        """Test that source formats are defined."""
        assert ColorSpace.RGB in SUPPORTED_SOURCE_FORMATS
        assert ColorSpace.BGR in SUPPORTED_SOURCE_FORMATS
        assert ColorSpace.RGBA in SUPPORTED_SOURCE_FORMATS
        assert ColorSpace.BGRA in SUPPORTED_SOURCE_FORMATS

    def test_yuyv_not_source(self):
        """Test that YUYV is not a source format."""
        assert ColorSpace.YUYV not in SUPPORTED_SOURCE_FORMATS

    def test_is_frozenset(self):
        """Test that SUPPORTED_SOURCE_FORMATS is immutable."""
        assert isinstance(SUPPORTED_SOURCE_FORMATS, frozenset)


class TestIsSupportedSource:
    """Tests for is_supported_source function."""

    def test_supported_source_formats(self):
        """Test supported source formats return True."""
        assert is_supported_source(ColorSpace.RGB)
        assert is_supported_source("rgb")
        assert is_supported_source(ColorSpace.BGR)
        assert is_supported_source("bgr")
        assert is_supported_source(ColorSpace.RGBA)
        assert is_supported_source("rgba")
        assert is_supported_source(ColorSpace.BGRA)
        assert is_supported_source("bgra")

    def test_unsupported_source_format(self):
        """Test unsupported source formats return False."""
        assert not is_supported_source(ColorSpace.YUYV)
        assert not is_supported_source("yuyv")
        assert not is_supported_source(ColorSpace.NV12)
        assert not is_supported_source("nv12")

    def test_invalid_source_format(self):
        """Test invalid format returns False."""
        assert not is_supported_source("invalid")
        assert not is_supported_source("xyz")

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert is_supported_source("RGB")
        assert is_supported_source("BGR")
        assert is_supported_source("RGBA")


class TestGetSourceChannels:
    """Tests for get_source_channels function."""

    def test_rgb_source_channels(self):
        """Test RGB source channels."""
        assert get_source_channels(ColorSpace.RGB) == 3
        assert get_source_channels("rgb") == 3

    def test_bgr_source_channels(self):
        """Test BGR source channels."""
        assert get_source_channels(ColorSpace.BGR) == 3
        assert get_source_channels("bgr") == 3

    def test_rgba_source_channels(self):
        """Test RGBA source channels."""
        assert get_source_channels(ColorSpace.RGBA) == 4
        assert get_source_channels("rgba") == 4

    def test_bgra_source_channels(self):
        """Test BGRA source channels."""
        assert get_source_channels(ColorSpace.BGRA) == 4
        assert get_source_channels("bgra") == 4

    def test_invalid_source_raises(self):
        """Test invalid source format raises ValueError."""
        with pytest.raises(ValueError, match="not a supported source format"):
            get_source_channels("yuyv")

    def test_unsupported_source_raises(self):
        """Test unsupported source format raises ValueError."""
        with pytest.raises(ValueError, match="not a supported source format"):
            get_source_channels(ColorSpace.YUYV)

    def test_case_insensitive(self):
        """Test case insensitivity."""
        assert get_source_channels("RGB") == 3
        assert get_source_channels("BGRA") == 4
