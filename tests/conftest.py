"""Pytest configuration and fixtures for camera adaptor tests."""

import numpy as np
import pytest


@pytest.fixture
def rgb_image():
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def rgb_image_float():
    """Create a sample RGB image as float32 for testing."""
    return np.random.rand(224, 224, 3).astype(np.float32)


@pytest.fixture
def rgba_image():
    """Create a sample RGBA image for testing."""
    return np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8)


@pytest.fixture
def batch_rgb_image():
    """Create a batch of RGB images for testing."""
    return np.random.rand(4, 224, 224, 3).astype(np.float32)


@pytest.fixture
def batch_rgba_image():
    """Create a batch of RGBA images for testing."""
    return np.random.rand(4, 224, 224, 4).astype(np.float32)


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "torch: marks tests as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "tensorflow: marks tests as requiring TensorFlow"
    )
    config.addinivalue_line(
        "markers", "lightning: marks tests as requiring PyTorch Lightning"
    )
    config.addinivalue_line(
        "markers", "opencv: marks tests as requiring OpenCV"
    )
