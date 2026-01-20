"""Tests for PyTorch Lightning integration."""

import pytest


@pytest.fixture(autouse=True)
def check_dependencies():
    """Skip all tests if dependencies are not available."""
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_lightning")


class TestCameraAdaptorCallback:
    """Tests for CameraAdaptorCallback."""

    def test_callback_creation(self):
        """Test creating callback."""
        from edgefirst.cameraadaptor.pytorch.lightning import (
            CameraAdaptorCallback,
        )

        callback = CameraAdaptorCallback("yuyv")
        assert callback.adaptor == "yuyv"

    def test_callback_factory(self):
        """Test create_callback factory function."""
        from edgefirst.cameraadaptor.pytorch.lightning import create_callback

        callback = create_callback("rgba")
        assert callback.adaptor == "rgba"


class TestCameraAdaptorMixin:
    """Tests for CameraAdaptorMixin."""

    def test_mixin_setup(self):
        """Test setting up camera adaptor with mixin."""
        pytest.importorskip("cv2")
        from edgefirst.cameraadaptor.pytorch.lightning import (
            CameraAdaptorMixin,
        )

        class TestModule(CameraAdaptorMixin):
            pass

        module = TestModule()
        module.setup_camera_adaptor("yuyv")

        assert module.camera_adaptor == "yuyv"
        assert module.camera_transform is not None
        assert module.camera_transform.channels == 2

    def test_mixin_config(self):
        """Test getting camera adaptor config from mixin."""
        pytest.importorskip("cv2")
        from edgefirst.cameraadaptor.pytorch.lightning import (
            CameraAdaptorMixin,
        )

        class TestModule(CameraAdaptorMixin):
            pass

        module = TestModule()
        module.setup_camera_adaptor("rgba")

        config = module.get_camera_adaptor_config()
        assert config["adaptor"] == "rgba"
        assert config["channels"] == 4
        assert config["output_channels"] == 3

    def test_mixin_no_setup(self):
        """Test mixin without setup returns empty config."""
        from edgefirst.cameraadaptor.pytorch.lightning import (
            CameraAdaptorMixin,
        )

        class TestModule(CameraAdaptorMixin):
            pass

        module = TestModule()
        config = module.get_camera_adaptor_config()
        assert config == {}
