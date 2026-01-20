"""Tests for transform module."""

import numpy as np
import pytest

from edgefirst.cameraadaptor import CameraAdaptorTransform, ColorSpace


class TestCameraAdaptorTransform:
    """Tests for CameraAdaptorTransform class."""

    def test_default_adaptor(self):
        """Test default adaptor is rgb."""
        transform = CameraAdaptorTransform()
        assert transform.adaptor == "rgb"
        assert transform.color_space == ColorSpace.RGB

    def test_default_source_format(self):
        """Test default source_format is rgb."""
        transform = CameraAdaptorTransform("yuyv")
        assert transform.source_format == "rgb"
        assert transform.source_color_space == ColorSpace.RGB

    def test_string_adaptor(self):
        """Test creating transform with string adaptor."""
        transform = CameraAdaptorTransform("yuyv")
        assert transform.adaptor == "yuyv"
        assert transform.color_space == ColorSpace.YUYV

    def test_case_insensitive(self):
        """Test case insensitivity."""
        transform = CameraAdaptorTransform("YUYV")
        assert transform.adaptor == "yuyv"

    def test_colorspace_adaptor(self):
        """Test creating transform with ColorSpace enum."""
        transform = CameraAdaptorTransform(ColorSpace.BGR)
        assert transform.adaptor == "bgr"
        assert transform.color_space == ColorSpace.BGR

    def test_invalid_adaptor(self):
        """Test invalid adaptor raises ValueError."""
        with pytest.raises(ValueError, match="Invalid target adaptor"):
            CameraAdaptorTransform("invalid")

    def test_unsupported_adaptor(self):
        """Test unsupported adaptor raises ValueError."""
        with pytest.raises(ValueError, match="not yet supported"):
            CameraAdaptorTransform("nv12")

    def test_channels_property(self):
        """Test channels property."""
        assert CameraAdaptorTransform("rgb").channels == 3
        assert CameraAdaptorTransform("rgba").channels == 4
        assert CameraAdaptorTransform("yuyv").channels == 2

    def test_input_channels_from_source(self):
        """Test input_channels reflects source format."""
        assert (
            CameraAdaptorTransform("yuyv", source_format="rgb").input_channels
            == 3
        )
        assert (
            CameraAdaptorTransform("yuyv", source_format="rgba").input_channels
            == 4
        )
        assert (
            CameraAdaptorTransform("yuyv", source_format="bgr").input_channels
            == 3
        )
        assert (
            CameraAdaptorTransform("yuyv", source_format="bgra").input_channels
            == 4
        )

    def test_output_channels(self):
        """Test output_channels property."""
        assert CameraAdaptorTransform("rgb").output_channels == 3
        assert CameraAdaptorTransform("rgba").output_channels == 3
        assert CameraAdaptorTransform("yuyv").output_channels == 2

    def test_str(self):
        """Test string representation."""
        transform = CameraAdaptorTransform("yuyv")
        assert str(transform) == "yuyv"

    def test_repr_default_source(self):
        """Test repr with default source format."""
        transform = CameraAdaptorTransform("yuyv")
        assert repr(transform) == "CameraAdaptorTransform(adaptor='yuyv')"

    def test_repr_custom_source(self):
        """Test repr with custom source format."""
        transform = CameraAdaptorTransform("yuyv", source_format="bgr")
        assert (
            repr(transform)
            == "CameraAdaptorTransform(adaptor='yuyv', source_format='bgr')"
        )

    def test_get_num_channels(self):
        """Test backward compatible get_num_channels method."""
        transform = CameraAdaptorTransform("rgba")
        assert transform.get_num_channels() == 4


class TestCameraAdaptorTransformSourceFormat:
    """Tests for source_format parameter."""

    def test_source_format_rgb(self):
        """Test explicit RGB source format."""
        transform = CameraAdaptorTransform("bgr", source_format="rgb")
        assert transform.source_format == "rgb"
        assert transform.source_color_space == ColorSpace.RGB

    def test_source_format_bgr(self):
        """Test BGR source format (OpenCV default)."""
        transform = CameraAdaptorTransform("yuyv", source_format="bgr")
        assert transform.source_format == "bgr"
        assert transform.source_color_space == ColorSpace.BGR

    def test_source_format_rgba(self):
        """Test RGBA source format."""
        transform = CameraAdaptorTransform("yuyv", source_format="rgba")
        assert transform.source_format == "rgba"
        assert transform.source_color_space == ColorSpace.RGBA

    def test_source_format_bgra(self):
        """Test BGRA source format."""
        transform = CameraAdaptorTransform("yuyv", source_format="bgra")
        assert transform.source_format == "bgra"
        assert transform.source_color_space == ColorSpace.BGRA

    def test_source_format_case_insensitive(self):
        """Test source format is case insensitive."""
        transform = CameraAdaptorTransform("yuyv", source_format="BGR")
        assert transform.source_format == "bgr"

    def test_invalid_source_format(self):
        """Test invalid source format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid source format"):
            CameraAdaptorTransform("yuyv", source_format="invalid")

    def test_unsupported_source_format(self):
        """Test unsupported source format raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            CameraAdaptorTransform("yuyv", source_format="yuyv")

    def test_colorspace_source_format(self):
        """Test ColorSpace enum as source format."""
        transform = CameraAdaptorTransform(
            "yuyv", source_format=ColorSpace.BGR
        )
        assert transform.source_format == "bgr"


@pytest.mark.opencv
class TestCameraAdaptorTransformConversions:
    """Tests for actual color conversions requiring OpenCV."""

    @pytest.fixture(autouse=True)
    def check_opencv(self):
        """Skip tests if OpenCV is not available."""
        pytest.importorskip("cv2")

    def test_rgb_identity(self, rgb_image):
        """Test RGB -> RGB is identity."""
        transform = CameraAdaptorTransform("rgb")
        result = transform(rgb_image)
        np.testing.assert_array_equal(result, rgb_image)

    def test_bgr_identity(self, rgb_image):
        """Test BGR -> BGR is identity."""
        import cv2

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        transform = CameraAdaptorTransform("bgr", source_format="bgr")
        result = transform(bgr_image)
        np.testing.assert_array_equal(result, bgr_image)

    def test_rgb_to_bgr_conversion(self, rgb_image):
        """Test RGB -> BGR conversion."""
        import cv2

        transform = CameraAdaptorTransform("bgr")
        result = transform(rgb_image)
        expected = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        np.testing.assert_array_equal(result, expected)

    def test_bgr_to_rgb_conversion(self, rgb_image):
        """Test BGR -> RGB conversion."""
        import cv2

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        transform = CameraAdaptorTransform("rgb", source_format="bgr")
        result = transform(bgr_image)
        np.testing.assert_array_equal(result, rgb_image)

    def test_rgba_conversion(self, rgb_image):
        """Test RGB -> RGBA conversion."""
        transform = CameraAdaptorTransform("rgba")
        result = transform(rgb_image)
        assert result.shape == (224, 224, 4)

    def test_bgra_conversion(self, rgb_image):
        """Test RGB -> BGRA conversion."""
        transform = CameraAdaptorTransform("bgra")
        result = transform(rgb_image)
        assert result.shape == (224, 224, 4)

    def test_yuyv_conversion(self, rgb_image):
        """Test RGB -> YUYV conversion."""
        transform = CameraAdaptorTransform("yuyv")
        result = transform(rgb_image)
        assert result.shape == (224, 224, 2)

    def test_bgr_to_yuyv_conversion(self, rgb_image):
        """Test BGR -> YUYV conversion."""
        import cv2

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        transform = CameraAdaptorTransform("yuyv", source_format="bgr")
        result = transform(bgr_image)
        assert result.shape == (224, 224, 2)

    def test_rgba_to_yuyv_conversion(self, rgba_image):
        """Test RGBA -> YUYV conversion."""
        transform = CameraAdaptorTransform("yuyv", source_format="rgba")
        result = transform(rgba_image)
        assert result.shape == (224, 224, 2)

    def test_callable_interface(self, rgb_image):
        """Test callable interface."""
        transform = CameraAdaptorTransform("bgr")
        result1 = transform(rgb_image)
        result2 = transform.convert(rgb_image)
        np.testing.assert_array_equal(result1, result2)

    def test_bgra_to_yuyv_conversion(self):
        """Test BGRA -> YUYV conversion."""

        # Create BGRA image
        bgra_image = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        transform = CameraAdaptorTransform("yuyv", source_format="bgra")
        result = transform(bgra_image)
        assert result.shape == (224, 224, 2)

    def test_rgba_to_rgb_conversion(self, rgba_image):
        """Test RGBA -> RGB conversion."""
        transform = CameraAdaptorTransform("rgb", source_format="rgba")
        result = transform(rgba_image)
        assert result.shape == (224, 224, 3)

    def test_bgra_to_rgb_conversion(self):
        """Test BGRA -> RGB conversion."""
        bgra_image = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        transform = CameraAdaptorTransform("rgb", source_format="bgra")
        result = transform(bgra_image)
        assert result.shape == (224, 224, 3)

    def test_bgra_to_bgr_conversion(self):
        """Test BGRA -> BGR conversion."""
        bgra_image = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        transform = CameraAdaptorTransform("bgr", source_format="bgra")
        result = transform(bgra_image)
        assert result.shape == (224, 224, 3)

    def test_rgba_to_bgr_conversion(self, rgba_image):
        """Test RGBA -> BGR conversion."""
        transform = CameraAdaptorTransform("bgr", source_format="rgba")
        result = transform(rgba_image)
        assert result.shape == (224, 224, 3)

    def test_rgba_to_bgra_conversion(self, rgba_image):
        """Test RGBA -> BGRA conversion."""
        transform = CameraAdaptorTransform("bgra", source_format="rgba")
        result = transform(rgba_image)
        assert result.shape == (224, 224, 4)

    def test_bgra_to_rgba_conversion(self):
        """Test BGRA -> RGBA conversion."""
        bgra_image = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        transform = CameraAdaptorTransform("rgba", source_format="bgra")
        result = transform(bgra_image)
        assert result.shape == (224, 224, 4)

    def test_yuyv_byte_order(self):
        """Verify YUYV produces correct byte order: Y0, U0, Y1, V0.

        This test ensures the OpenCV conversion produces true YUYV format,
        not YVYU (which would swap U and V positions). The byte order is
        critical for correct color reproduction at inference when the
        camera provides YUYV data.

        YUYV (also called YUY2): Y0 U0 Y1 V0 (4:2:2 packed)
        YVYU: Y0 V0 Y1 U0 (different packed format)
        """
        import cv2

        # Use known RGB values: Red and Green pixels
        # Red (255,0,0): Y≈82, U≈90, V≈240 (in BT.601)
        # Green (0,255,0): Y≈145, U≈54, V≈34 (in BT.601)
        rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)

        transform = CameraAdaptorTransform("yuyv")
        yuyv = transform(rgb)

        # Verify shape: 2 pixels pack into (H, W, 2) with 4 bytes per pair
        assert yuyv.shape == (1, 2, 2), f"Expected (1, 2, 2), got {yuyv.shape}"

        # Verify it matches OpenCV's YUYV output (not YVYU)
        expected_yuyv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_YUYV)
        np.testing.assert_array_equal(yuyv, expected_yuyv)

        # Cross-check: verify it's NOT the YVYU format
        wrong_yvyu = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_YVYU)
        assert not np.array_equal(yuyv, wrong_yvyu), (
            "Output matches YVYU, not YUYV!"
        )

        # Verify byte order by checking Y values (stable across YUV variants)
        # Y for red should be lower (~82) than Y for green (~145)
        y0 = yuyv[0, 0, 0]  # Y for first pixel (red)
        y1 = yuyv[0, 1, 0]  # Y for second pixel (green)
        assert 75 <= y0 <= 90, f"Y for red should be ~82, got {y0}"
        assert 140 <= y1 <= 155, f"Y for green should be ~145, got {y1}"

    def test_yuyv_roundtrip_preserves_colors(self):
        """Test that YUYV conversion and back preserves colors reasonably.

        Note: YUV 4:2:2 subsampling is lossy for chroma, so we allow
        some tolerance. This test verifies the conversion pipeline
        produces valid, decodable YUYV data.
        """
        import cv2

        # Use a simple RGB image with solid colors
        rgb = np.array(
            [
                [[255, 0, 0], [255, 0, 0]],  # Two red pixels
                [[0, 255, 0], [0, 255, 0]],  # Two green pixels
            ],
            dtype=np.uint8,
        )

        transform = CameraAdaptorTransform("yuyv")
        yuyv = transform(rgb)

        # Decode YUYV back to RGB
        rgb_back = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)

        # Red pixels should still be reddish (R > G and R > B)
        assert rgb_back[0, 0, 0] > rgb_back[0, 0, 1], (
            "Red channel should dominate for red"
        )
        assert rgb_back[0, 0, 0] > rgb_back[0, 0, 2], (
            "Red channel should dominate for red"
        )

        # Green pixels should still be greenish (G > R and G > B)
        assert rgb_back[1, 0, 1] > rgb_back[1, 0, 0], (
            "Green channel should dominate for green"
        )
        assert rgb_back[1, 0, 1] > rgb_back[1, 0, 2], (
            "Green channel should dominate for green"
        )
