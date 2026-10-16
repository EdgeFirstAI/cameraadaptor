"""TensorFlow/Keras support for camera adaptor.

Tiled training and validation ops are in ``edgefirst.cameraadaptor.tensorflow
.tiling`` (TILING.md §7.4).
"""

from .adaptor import CameraAdaptor

__all__ = ["CameraAdaptor"]
