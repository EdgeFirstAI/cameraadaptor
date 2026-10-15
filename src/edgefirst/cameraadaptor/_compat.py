"""Compatibility utilities for edgefirst-cameraadaptor."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Python version checks
PY38 = sys.version_info >= (3, 8)
PY39 = sys.version_info >= (3, 9)
PY310 = sys.version_info >= (3, 10)


def _installed(module: str) -> bool:
    # Importing a framework only to test for it is slow and, for some
    # TensorFlow and PyTorch Lightning builds, crashes when both load in one
    # process.
    return importlib.util.find_spec(module) is not None


def check_torch_available() -> bool:
    """Check if PyTorch is installed, without importing it."""
    return _installed("torch")


def check_tensorflow_available() -> bool:
    """Check if TensorFlow is installed, without importing it."""
    return _installed("tensorflow")


def check_lightning_available() -> bool:
    """Check if PyTorch Lightning is installed, without importing it."""
    return _installed("pytorch_lightning")


def check_opencv_available() -> bool:
    """Check if OpenCV is installed, without importing it."""
    return _installed("cv2")


def require_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not check_torch_available():
        raise ImportError(
            "PyTorch is required for this functionality. "
            "Install with: pip install edgefirst-cameraadaptor[torch]"
        )


def require_tensorflow() -> None:
    """Raise ImportError if TensorFlow is not available."""
    if not check_tensorflow_available():
        raise ImportError(
            "TensorFlow is required for this functionality. "
            "Install with: pip install edgefirst-cameraadaptor[tensorflow]"
        )


def require_lightning() -> None:
    """Raise ImportError if PyTorch Lightning is not available."""
    if not check_lightning_available():
        raise ImportError(
            "PyTorch Lightning is required for this functionality. "
            "Install with: pip install edgefirst-cameraadaptor[lightning]"
        )


def require_opencv() -> None:
    """Raise ImportError if OpenCV is not available."""
    if not check_opencv_available():
        raise ImportError(
            "OpenCV is required for this functionality. "
            "Install with: pip install edgefirst-cameraadaptor[transform]"
        )
