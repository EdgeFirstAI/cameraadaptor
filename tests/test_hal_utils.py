"""Tests for HAL utilities module."""

import pytest

from edgefirst.cameraadaptor import ColorSpace
from edgefirst.cameraadaptor.hal_utils import (
    COLORSPACE_MAP,
    FOURCC_MAP,
    HAL_FOURCC_ALIASES,
    PLANAR_FOURCC_MAP,
    from_fourcc_str,
    get_fourcc_str,
    get_planar_fourcc_str,
    resolve_hal_alias,
)


class TestFourCCMappings:
    """Tests for FourCC mapping dictionaries."""

    def test_fourcc_map_contains_supported_formats(self):
        """Test FOURCC_MAP contains all currently supported formats."""
        assert ColorSpace.RGB in FOURCC_MAP
        assert ColorSpace.BGR in FOURCC_MAP
        assert ColorSpace.RGBA in FOURCC_MAP
        assert ColorSpace.BGRA in FOURCC_MAP
        assert ColorSpace.GREY in FOURCC_MAP
        assert ColorSpace.YUYV in FOURCC_MAP

    def test_fourcc_map_values(self):
        """Test FOURCC_MAP has correct FourCC codes."""
        assert FOURCC_MAP[ColorSpace.RGB] == "RGB3"
        assert FOURCC_MAP[ColorSpace.BGR] == "BGR3"
        assert FOURCC_MAP[ColorSpace.RGBA] == "RGBA"
        assert FOURCC_MAP[ColorSpace.BGRA] == "BGRA"
        assert FOURCC_MAP[ColorSpace.GREY] == "GREY"
        assert FOURCC_MAP[ColorSpace.YUYV] == "YUYV"

    def test_planar_fourcc_map_values(self):
        """Test PLANAR_FOURCC_MAP has correct planar FourCC codes."""
        assert PLANAR_FOURCC_MAP[ColorSpace.RGB] == "RGBP"
        assert PLANAR_FOURCC_MAP[ColorSpace.BGR] == "BGRP"
        assert PLANAR_FOURCC_MAP[ColorSpace.RGBA] == "RGBP"
        assert PLANAR_FOURCC_MAP[ColorSpace.BGRA] == "BGRP"

    def test_colorspace_map_is_reverse_of_fourcc_map(self):
        """Test COLORSPACE_MAP is the reverse of FOURCC_MAP."""
        for cs, fourcc in FOURCC_MAP.items():
            assert COLORSPACE_MAP[fourcc] == cs

    def test_hal_fourcc_aliases_contains_planar_formats(self):
        """Test HAL_FOURCC_ALIASES contains planar format aliases."""
        assert "PLANAR_RGB" in HAL_FOURCC_ALIASES
        assert "PLANAR_RGBA" in HAL_FOURCC_ALIASES
        assert "PLANAR_BGR" in HAL_FOURCC_ALIASES
        assert "PLANAR_BGRA" in HAL_FOURCC_ALIASES

    def test_hal_fourcc_aliases_contains_yuv_formats(self):
        """Test HAL_FOURCC_ALIASES contains YUV format aliases."""
        assert "NV16" in HAL_FOURCC_ALIASES
        assert "NV12" in HAL_FOURCC_ALIASES
        assert "NV21" in HAL_FOURCC_ALIASES


class TestGetFourCCStr:
    """Tests for get_fourcc_str function."""

    def test_get_fourcc_str_from_colorspace(self):
        """Test get_fourcc_str with ColorSpace enum."""
        assert get_fourcc_str(ColorSpace.YUYV) == "YUYV"
        assert get_fourcc_str(ColorSpace.RGB) == "RGB3"
        assert get_fourcc_str(ColorSpace.BGR) == "BGR3"

    def test_get_fourcc_str_from_string(self):
        """Test get_fourcc_str with string input."""
        assert get_fourcc_str("yuyv") == "YUYV"
        assert get_fourcc_str("rgb") == "RGB3"
        assert get_fourcc_str("RGBA") == "RGBA"

    def test_get_fourcc_str_invalid(self):
        """Test get_fourcc_str raises for invalid input."""
        with pytest.raises(ValueError, match="Invalid color space"):
            get_fourcc_str("invalid")


class TestGetPlanarFourCCStr:
    """Tests for get_planar_fourcc_str function."""

    def test_get_planar_fourcc_str_rgb(self):
        """Test get_planar_fourcc_str for RGB."""
        assert get_planar_fourcc_str("rgb") == "RGBP"
        assert get_planar_fourcc_str(ColorSpace.RGB) == "RGBP"

    def test_get_planar_fourcc_str_bgr(self):
        """Test get_planar_fourcc_str for BGR."""
        assert get_planar_fourcc_str("bgr") == "BGRP"
        assert get_planar_fourcc_str(ColorSpace.BGR) == "BGRP"

    def test_get_planar_fourcc_str_rgba(self):
        """Test get_planar_fourcc_str for RGBA."""
        assert get_planar_fourcc_str("rgba") == "RGBP"
        assert get_planar_fourcc_str(ColorSpace.RGBA) == "RGBP"

    def test_get_planar_fourcc_str_bgra(self):
        """Test get_planar_fourcc_str for BGRA."""
        assert get_planar_fourcc_str("bgra") == "BGRP"
        assert get_planar_fourcc_str(ColorSpace.BGRA) == "BGRP"

    def test_get_planar_fourcc_str_no_planar_format(self):
        """Test get_planar_fourcc_str raises for non-planar formats."""
        with pytest.raises(ValueError, match="No planar FourCC mapping"):
            get_planar_fourcc_str("yuyv")

    def test_get_planar_fourcc_str_invalid(self):
        """Test get_planar_fourcc_str raises for invalid input."""
        with pytest.raises(ValueError, match="Invalid color space"):
            get_planar_fourcc_str("invalid")


class TestFromFourCCStr:
    """Tests for from_fourcc_str function."""

    def test_from_fourcc_str_yuyv(self):
        """Test from_fourcc_str for YUYV."""
        assert from_fourcc_str("YUYV") == ColorSpace.YUYV

    def test_from_fourcc_str_rgb3(self):
        """Test from_fourcc_str for RGB3."""
        assert from_fourcc_str("RGB3") == ColorSpace.RGB

    def test_from_fourcc_str_case_insensitive(self):
        """Test from_fourcc_str is case-insensitive."""
        assert from_fourcc_str("yuyv") == ColorSpace.YUYV
        assert from_fourcc_str("rgb3") == ColorSpace.RGB

    def test_from_fourcc_str_invalid(self):
        """Test from_fourcc_str raises for invalid input."""
        with pytest.raises(ValueError, match="Unknown FourCC code"):
            from_fourcc_str("INVALID")


class TestResolveHalAlias:
    """Tests for resolve_hal_alias function."""

    def test_resolve_planar_rgb(self):
        """Test resolve_hal_alias for PLANAR_RGB."""
        assert resolve_hal_alias("PLANAR_RGB") == "RGBP"
        assert resolve_hal_alias("planar_rgb") == "RGBP"

    def test_resolve_planar_rgba(self):
        """Test resolve_hal_alias for PLANAR_RGBA."""
        assert resolve_hal_alias("PLANAR_RGBA") == "RGBP"

    def test_resolve_planar_bgr(self):
        """Test resolve_hal_alias for PLANAR_BGR."""
        assert resolve_hal_alias("PLANAR_BGR") == "BGRP"

    def test_resolve_planar_bgra(self):
        """Test resolve_hal_alias for PLANAR_BGRA."""
        assert resolve_hal_alias("PLANAR_BGRA") == "BGRP"

    def test_resolve_nv16(self):
        """Test resolve_hal_alias for NV16."""
        assert resolve_hal_alias("NV16") == "NV16"
        assert resolve_hal_alias("nv16") == "NV16"

    def test_resolve_nv12(self):
        """Test resolve_hal_alias for NV12."""
        assert resolve_hal_alias("NV12") == "NV12"

    def test_resolve_nv21(self):
        """Test resolve_hal_alias for NV21."""
        assert resolve_hal_alias("NV21") == "NV21"

    def test_resolve_existing_fourcc_passthrough(self):
        """Test resolve_hal_alias passes through existing FourCC codes."""
        assert resolve_hal_alias("YUYV") == "YUYV"
        assert resolve_hal_alias("RGB3") == "RGB3"

    def test_resolve_hal_alias_invalid(self):
        """Test resolve_hal_alias raises for invalid input."""
        with pytest.raises(ValueError, match="Unknown HAL FourCC alias"):
            resolve_hal_alias("INVALID")
