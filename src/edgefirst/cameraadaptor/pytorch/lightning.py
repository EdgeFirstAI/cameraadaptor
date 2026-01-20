"""PyTorch Lightning integration for camera adaptor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer


class CameraAdaptorCallback:
    """PyTorch Lightning callback for logging camera adaptor configuration.

    This callback logs the camera adaptor configuration to the experiment
    tracker (e.g., TensorBoard, Weights & Biases, MLflow) at the start
    of training.

    Args:
        adaptor: The camera adaptor name or configuration.
        log_to_hparams: Whether to log to hyperparameters. Defaults to True.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from edgefirst.cameraadaptor.pytorch.lightning import (
        ...     CameraAdaptorCallback,
        ... )
        >>> callback = CameraAdaptorCallback("yuyv")
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        adaptor: str,
        log_to_hparams: bool = True,
    ) -> None:
        import importlib.util

        if importlib.util.find_spec("pytorch_lightning") is None:
            raise ImportError(
                "PyTorch Lightning is required for this callback. "
                "Install with: pip install edgefirst-camera-adaptor[lightning]"
            )

        self._adaptor = adaptor
        self._log_to_hparams = log_to_hparams

    @property
    def adaptor(self) -> str:
        """Get the adaptor name."""
        return self._adaptor

    def on_fit_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Log camera adaptor configuration at the start of training.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module being trained.
        """
        if self._log_to_hparams and trainer.logger is not None:
            trainer.logger.log_hyperparams({"camera_adaptor": self._adaptor})


class CameraAdaptorMixin:
    """Mixin for LightningDataModule to add camera adaptor support.

    This mixin provides camera adaptor functionality to a LightningDataModule,
    making it easy to integrate camera format conversion into data pipelines.

    Example:
        >>> from pytorch_lightning import LightningDataModule
        >>> from edgefirst.cameraadaptor.pytorch.lightning import (
        ...     CameraAdaptorMixin,
        ... )
        >>> from edgefirst.cameraadaptor import CameraAdaptorTransform
        >>>
        >>> class MyDataModule(CameraAdaptorMixin, LightningDataModule):
        ...     def __init__(self, adaptor: str = "rgb"):
        ...         super().__init__()
        ...         self.setup_camera_adaptor(adaptor)
        ...
        ...     def train_dataloader(self):
        ...         # Use self.camera_transform in your data pipeline
        ...         ...
    """

    _camera_adaptor: str | None = None
    _camera_transform: Any | None = None

    def setup_camera_adaptor(self, adaptor: str = "rgb") -> None:
        """Set up the camera adaptor for this data module.

        Args:
            adaptor: Target color space name. Defaults to "rgb".
        """
        from ..transform import CameraAdaptorTransform

        self._camera_adaptor = adaptor
        self._camera_transform = CameraAdaptorTransform(adaptor)

    @property
    def camera_adaptor(self) -> str | None:
        """Get the camera adaptor name."""
        return self._camera_adaptor

    @property
    def camera_transform(self) -> Any | None:
        """Get the camera transform instance."""
        return self._camera_transform

    def get_camera_adaptor_config(self) -> dict[str, Any]:
        """Get camera adaptor configuration as a dictionary.

        Returns:
            Dictionary with camera adaptor configuration.
        """
        if self._camera_transform is None:
            return {}

        return {
            "adaptor": self._camera_adaptor,
            "channels": self._camera_transform.channels,
            "output_channels": self._camera_transform.output_channels,
        }


# Create actual callback class that inherits from Lightning Callback
def create_callback(adaptor: str, log_to_hparams: bool = True) -> Any:
    """Create a PyTorch Lightning callback for camera adaptor logging.

    This factory function creates a proper Lightning Callback instance.

    Args:
        adaptor: The camera adaptor name.
        log_to_hparams: Whether to log to hyperparameters. Defaults to True.

    Returns:
        A PyTorch Lightning Callback instance.
    """
    try:
        from pytorch_lightning.callbacks import Callback
    except ImportError:
        raise ImportError(
            "PyTorch Lightning is required for this callback. "
            "Install with: pip install edgefirst-camera-adaptor[lightning]"
        ) from None

    class _CameraAdaptorCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self._adaptor = adaptor
            self._log_to_hparams = log_to_hparams

        @property
        def adaptor(self) -> str:
            return self._adaptor

        def on_fit_start(
            self, trainer: Trainer, pl_module: LightningModule
        ) -> None:
            if self._log_to_hparams and trainer.logger is not None:
                trainer.logger.log_hyperparams(
                    {"camera_adaptor": self._adaptor}
                )

    return _CameraAdaptorCallback()


__all__ = ["CameraAdaptorCallback", "CameraAdaptorMixin", "create_callback"]
