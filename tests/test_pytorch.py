"""Tests for PyTorch support."""

import pytest


@pytest.fixture(autouse=True)
def check_torch():
    """Skip all tests in this module if PyTorch is not available."""
    pytest.importorskip("torch")


class TestCameraAdaptorModule:
    """Tests for PyTorch CameraAdaptor module."""

    def test_default_adaptor(self):
        """Test default adaptor is rgb."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor()
        assert adaptor.adaptor == "rgb"

    def test_default_channels_last(self):
        """Test default channels_last is False."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor()
        assert adaptor.channels_last is False

    def test_string_adaptor(self):
        """Test creating adaptor with string."""
        from edgefirst.cameraadaptor import ColorSpace
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("yuyv")
        assert adaptor.adaptor == "yuyv"
        assert adaptor.color_space == ColorSpace.YUYV

    def test_colorspace_adaptor(self):
        """Test creating adaptor with ColorSpace enum."""
        from edgefirst.cameraadaptor import ColorSpace
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor(ColorSpace.BGR)
        assert adaptor.adaptor == "bgr"

    def test_invalid_adaptor(self):
        """Test invalid adaptor raises ValueError."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        with pytest.raises(ValueError, match="Invalid camera adaptor"):
            CameraAdaptor("invalid")

    def test_unsupported_adaptor(self):
        """Test unsupported adaptor raises ValueError."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        with pytest.raises(ValueError, match="not yet supported"):
            CameraAdaptor("nv12")

    def test_forward_rgb(self):
        """Test forward pass with RGB input."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgb")
        x = torch.randn(1, 3, 224, 224)
        y = adaptor(x)
        assert y.shape == (1, 3, 224, 224)
        assert torch.equal(x, y)

    def test_forward_rgba_drops_alpha(self):
        """Test forward pass with RGBA drops alpha channel."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgba")
        x = torch.randn(1, 4, 224, 224)
        y = adaptor(x)
        assert y.shape == (1, 3, 224, 224)
        # First 3 channels should match
        assert torch.equal(x[:, :3, :, :], y)

    def test_forward_bgra_drops_alpha(self):
        """Test forward pass with BGRA drops alpha channel."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("bgra")
        x = torch.randn(1, 4, 224, 224)
        y = adaptor(x)
        assert y.shape == (1, 3, 224, 224)

    def test_forward_yuyv(self):
        """Test forward pass with YUYV input."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("yuyv")
        x = torch.randn(1, 2, 224, 224)
        y = adaptor(x)
        assert y.shape == (1, 2, 224, 224)
        assert torch.equal(x, y)

    def test_compute_input_channels(self):
        """Test compute_input_channels static method."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        assert CameraAdaptor.compute_input_channels("rgb") == 3
        assert CameraAdaptor.compute_input_channels("rgba") == 4
        assert CameraAdaptor.compute_input_channels("yuyv") == 2
        assert CameraAdaptor.compute_input_channels(["rgba"]) == 4

    def test_compute_output_channels(self):
        """Test compute_output_channels static method."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        assert CameraAdaptor.compute_output_channels("rgb") == 3
        assert CameraAdaptor.compute_output_channels("rgba") == 3
        assert CameraAdaptor.compute_output_channels("yuyv") == 2
        assert CameraAdaptor.compute_output_channels(["bgra"]) == 3

    def test_extra_repr(self):
        """Test extra_repr method."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("yuyv")
        assert "yuyv" in adaptor.extra_repr()

    def test_extra_repr_channels_last(self):
        """Test extra_repr includes channels_last when True."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("yuyv", channels_last=True)
        repr_str = adaptor.extra_repr()
        assert "yuyv" in repr_str
        assert "channels_last=True" in repr_str

    def test_module_is_trainable(self):
        """Test that module can be part of training."""
        import torch
        import torch.nn as nn

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptor = CameraAdaptor("rgba")
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                x = self.adaptor(x)
                return self.conv(x)

        model = TestModel()
        x = torch.randn(1, 4, 32, 32)
        y = model(x)
        assert y.shape == (1, 16, 32, 32)


class TestCameraAdaptorChannelsLast:
    """Tests for channels_last parameter."""

    def test_channels_last_property(self):
        """Test channels_last property."""
        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgb", channels_last=True)
        assert adaptor.channels_last is True

    def test_forward_channels_last_rgb(self):
        """Test forward pass with channels-last RGB input."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgb", channels_last=True)
        x = torch.randn(1, 224, 224, 3)  # NHWC
        y = adaptor(x)
        assert y.shape == (1, 3, 224, 224)  # NCHW output

    def test_forward_channels_last_yuyv(self):
        """Test forward pass with channels-last YUYV input."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("yuyv", channels_last=True)
        x = torch.randn(1, 224, 224, 2)  # NHWC
        y = adaptor(x)
        assert y.shape == (1, 2, 224, 224)  # NCHW output

    def test_forward_channels_last_rgba(self):
        """Test forward pass with channels-last RGBA drops alpha."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgba", channels_last=True)
        x = torch.randn(1, 224, 224, 4)  # NHWC
        y = adaptor(x)
        assert y.shape == (1, 3, 224, 224)  # NCHW output, alpha dropped

    def test_channels_last_permute_correctness(self):
        """Test that channels-last permute is correct."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgb", channels_last=True)

        # Create tensor with known values
        x = torch.arange(12).reshape(1, 2, 2, 3).float()  # NHWC
        y = adaptor(x)

        # Verify shape
        assert y.shape == (1, 3, 2, 2)  # NCHW

        # Verify values are correctly permuted
        # NHWC[0,0,0,c] -> NCHW[0,c,0,0]
        for c in range(3):
            assert y[0, c, 0, 0] == x[0, 0, 0, c]

    def test_module_with_channels_last_in_model(self):
        """Test channels_last adaptor in a model."""
        import torch
        import torch.nn as nn

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaptor = CameraAdaptor("rgba", channels_last=True)
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                x = self.adaptor(x)  # NHWC -> NCHW, drop alpha
                return self.conv(x)

        model = TestModel()
        x = torch.randn(1, 32, 32, 4)  # NHWC input
        y = model(x)
        assert y.shape == (1, 16, 32, 32)


class TestCameraAdaptorBatchProcessing:
    """Tests for batch processing."""

    def test_batch_processing(self):
        """Test processing batches of images."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgba")
        batch_size = 8
        x = torch.randn(batch_size, 4, 224, 224)
        y = adaptor(x)
        assert y.shape == (batch_size, 3, 224, 224)

    def test_batch_processing_channels_last(self):
        """Test batch processing with channels_last."""
        import torch

        from edgefirst.cameraadaptor.pytorch import CameraAdaptor

        adaptor = CameraAdaptor("rgba", channels_last=True)
        batch_size = 8
        x = torch.randn(batch_size, 224, 224, 4)  # NHWC
        y = adaptor(x)
        assert y.shape == (batch_size, 3, 224, 224)  # NCHW
