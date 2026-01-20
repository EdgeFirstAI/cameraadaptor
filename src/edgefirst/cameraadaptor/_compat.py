"""Compatibility utilities for edgefirst-camera-adaptor."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Python version checks
PY38 = sys.version_info >= (3, 8)
PY39 = sys.version_info >= (3, 9)
PY310 = sys.version_info >= (3, 10)


def check_torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def check_tensorflow_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def check_lightning_available() -> bool:
    """Check if PyTorch Lightning is available."""
    try:
        import pytorch_lightning  # noqa: F401

        return True
    except ImportError:
        return False


def check_opencv_available() -> bool:
    """Check if OpenCV is available."""
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def require_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not check_torch_available():
        raise ImportError(
            "PyTorch is required for this functionality. "
            "Install with: pip install edgefirst-camera-adaptor[torch]"
        )


def require_tensorflow() -> None:
    """Raise ImportError if TensorFlow is not available."""
    if not check_tensorflow_available():
        raise ImportError(
            "TensorFlow is required for this functionality. "
            "Install with: pip install edgefirst-camera-adaptor[tensorflow]"
        )


def require_lightning() -> None:
    """Raise ImportError if PyTorch Lightning is not available."""
    if not check_lightning_available():
        raise ImportError(
            "PyTorch Lightning is required for this functionality. "
            "Install with: pip install edgefirst-camera-adaptor[lightning]"
        )


def require_opencv() -> None:
    """Raise ImportError if OpenCV is not available."""
    if not check_opencv_available():
        raise ImportError(
            "OpenCV is required for this functionality. "
            "Install with: pip install edgefirst-camera-adaptor[transform]"
        )
