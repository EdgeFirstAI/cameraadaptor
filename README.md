# EdgeFirst CameraAdaptor

Train deep learning models that consume camera formats natively supported by target edge platforms, avoiding costly runtime conversions.

## Why CameraAdaptor?

When deploying computer vision models to edge devices, there's often a mismatch between:

1. **Training data format**: RGB images from standard datasets (ImageNet, COCO, etc.)
2. **Inference input format**: Native camera/hardware formats (YUV, Bayer, BGR, RGBA)

**Traditional approach**: Convert camera output → RGB → Model inference

**Problem**: This conversion requires hardware (ISP, GPU, 2D accelerator) and adds latency.

**Solution**: Train the model to expect the native camera format directly.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING TIME                            │
├─────────────────────────────────────────────────────────────────┤
│  RGB Dataset ─── CameraAdaptorTransform ──→ Target Format       │
│                        (e.g., YUYV, BGR)                        │
│                              ↓                                  │
│                    Model with CameraAdaptor                     │
│                    as first layer                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE TIME                           │
├─────────────────────────────────────────────────────────────────┤
│  Camera/Hardware ────────────────────────→ Model (native format)│
│                    No conversion needed!                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `CameraAdaptorTransform` | Preprocessing: Convert RGB training data to target format |
| `CameraAdaptor` (PyTorch) | Model layer: Handle format-specific input processing |
| `CameraAdaptor` (TensorFlow) | Model layer: Handle format-specific input processing |
| `CameraAdaptorConfig` | Configuration and metadata for model export |

### Important Design Note

The `CameraAdaptor` layer does **NOT** perform color space conversion. Color conversion
is handled by `CameraAdaptorTransform` during training data loading:

- **Training**: `CameraAdaptorTransform` converts RGB images → target format (e.g., YUYV, BGR)
- **Inference**: Camera/ISP provides data directly in target format → no conversion needed

The `CameraAdaptor` layer only performs:
1. Layout permutation (NHWC ↔ NCHW) when `channels_last`/`channels_first` is enabled
2. Alpha channel dropping for RGBA/BGRA inputs

## EdgeFirst Ecosystem

EdgeFirst CameraAdaptor is part of the EdgeFirst AI ecosystem:

- **[EdgeFirst HAL](https://github.com/EdgeFirstAI/hal)**: Runtime library with optimized pre-processing pipelines for edge deployment. Use HAL for on-target inference and benchmarking of models trained with CameraAdaptor.
- **EdgeFirst CameraAdaptor**: Training library (this project) for creating models that accept native camera formats.

On-target benchmarks use `edgefirst-hal` to benchmark pre-processing pipelines with various CameraAdaptor configurations.

## Installation

```bash
# Core library (numpy only)
pip install edgefirst-cameraadaptor

# With preprocessing support (OpenCV)
pip install edgefirst-cameraadaptor[transform]

# With PyTorch support
pip install edgefirst-cameraadaptor[torch]

# With TensorFlow support
pip install edgefirst-cameraadaptor[tensorflow]

# With PyTorch Lightning support
pip install edgefirst-cameraadaptor[lightning]

# Everything
pip install edgefirst-cameraadaptor[all]
```

## Quick Start

### Preprocessing Transform

Convert training images to your target camera format:

```python
from edgefirst.cameraadaptor import CameraAdaptorTransform

# Create transform for BGR format (RGB source by default)
transform = CameraAdaptorTransform("bgr")
bgr_frame = transform(rgb_frame)

# If using OpenCV's default BGR loading
transform = CameraAdaptorTransform("yuyv", source_format="bgr")
yuyv_frame = transform(bgr_frame)  # cv2.imread() returns BGR
```

### PyTorch Model

Add the adaptor as the first layer of your model:

```python
from edgefirst.cameraadaptor.pytorch import CameraAdaptor
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, adaptor="rgb"):
        super().__init__()
        self.adaptor = CameraAdaptor(adaptor)
        self.backbone = nn.Sequential(
            nn.Conv2d(CameraAdaptor.compute_output_channels(adaptor), 64, 3),
            # ... rest of your model
        )

    def forward(self, x):
        x = self.adaptor(x)
        return self.backbone(x)

# Model for RGBA input (4 channels -> 3 channels after adaptor)
model = MyModel(adaptor="rgba")
```

### TensorFlow/Keras Model

```python
from edgefirst.cameraadaptor.tensorflow import CameraAdaptor
import tensorflow as tf

inputs = tf.keras.Input(shape=(224, 224, 4))  # RGBA input
x = CameraAdaptor("rgba")(inputs)  # Drops alpha -> 3 channels
x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
# ... rest of your model
```

### Channels-Last Input (Camera Pipeline Direct)

For models receiving data directly from camera pipelines in NHWC format:

```python
# PyTorch: accept channels-last input, convert to channels-first internally
adaptor = CameraAdaptor("yuyv", channels_last=True)
x = torch.randn(1, 224, 224, 2)  # NHWC from camera
y = adaptor(x)  # Output: (1, 2, 224, 224) in NCHW

# TensorFlow: accept channels-first input if needed
from edgefirst.cameraadaptor.tensorflow import CameraAdaptor
layer = CameraAdaptor("yuyv", channels_first=True)
x = tf.random.normal((1, 2, 224, 224))  # NCHW
y = layer(x)  # Output: (1, 224, 224, 2) in NHWC
```

### Ultralytics YAML Configuration

```yaml
# YOLOv8 model with RGBA input
backbone:
  - [-1, 1, CameraAdaptor, [rgba]]  # First layer
  - [-1, 1, Conv, [64, 3, 2]]
  # ... rest of backbone
```

## Source Format (Data Loader Compatibility)

Different image loading libraries return different formats:

| Library | Default Format | Transform Setup |
|---------|---------------|--------------------|
| PIL/Pillow | RGB | `source_format="rgb"` (default) |
| torchvision | RGB | `source_format="rgb"` (default) |
| OpenCV cv2.imread() | **BGR** | `source_format="bgr"` |
| OpenCV cv2.IMREAD_UNCHANGED | BGRA | `source_format="bgra"` |
| imageio | RGB | `source_format="rgb"` (default) |
| skimage | RGB | `source_format="rgb"` (default) |

**Important:** OpenCV loads images as BGR by default. If you're using `cv2.imread()` without explicit conversion, set `source_format="bgr"`:

```python
import cv2
from edgefirst.cameraadaptor import CameraAdaptorTransform

# CORRECT: Tell the transform your source is BGR
img = cv2.imread("image.jpg")
transform = CameraAdaptorTransform("yuyv", source_format="bgr")
yuyv = transform(img)
```

## Supported Color Spaces

### Currently Supported

| Format | Input Channels | Output Channels | Description |
|--------|---------------|-----------------|-------------|
| RGB | 3 | 3 | Standard RGB |
| BGR | 3 | 3 | OpenCV native |
| RGBA | 4 | 3 | RGB + alpha (dropped) |
| BGRA | 4 | 3 | BGR + alpha (dropped) |
| YUYV | 2 | 2 | YUV 4:2:2, ch0=Y, ch1=UV |

### Planned

- **Roadmap**: NV12, NV21 (semi-planar YUV 4:2:0)
- **Roadmap**: Bayer patterns (RGGB, BGGR, GRBG, GBRG)

See [FORMATS.md](FORMATS.md) for detailed format documentation.

## Platform-Specific Guidance

See [PLATFORMS.md](PLATFORMS.md) for i.MX platform-specific recommendations:

- **i.MX 93**: PXP outputs BGR - train models with BGR format
- **i.MX 8M Plus**: G2D outputs RGBA - use RGBA to auto-slice alpha
- **i.MX 95**: ISI/ISP pipeline considerations

## Configuration

Use `CameraAdaptorConfig` for model metadata:

```python
from edgefirst.cameraadaptor import CameraAdaptorConfig

config = CameraAdaptorConfig(
    adaptor="yuyv",
    input_dtype="uint8",   # For quantized models
    output_dtype="uint8",
)

# Embed in model metadata
metadata = config.to_metadata()
```

## PyTorch Lightning Integration

```python
from pytorch_lightning import Trainer
from edgefirst.cameraadaptor.pytorch.lightning import create_callback

callback = create_callback("yuyv")
trainer = Trainer(callbacks=[callback])
```

## Migration from Existing Code

### From ultralytics/edgefirst

```python
# Before
from ultralytics.edgefirst.camera.adaptor import CameraAdaptorTransform
from ultralytics.edgefirst.nn.modules import CameraAdaptor

# After
from edgefirst.cameraadaptor import CameraAdaptorTransform
from edgefirst.cameraadaptor.pytorch import CameraAdaptor
```

### From modelpack

```python
# Before
from deepview.modelpack.datasets.color import ColorAdaptor
from deepview.modelpack.layers.conv2d import ColorAdaptor as TFColorAdaptor

# After
from edgefirst.cameraadaptor import CameraAdaptorTransform
from edgefirst.cameraadaptor.tensorflow import CameraAdaptor
```

## API Reference

### `CameraAdaptorTransform`

Preprocessing transform for converting images to target formats.

```python
transform = CameraAdaptorTransform(
    adaptor="yuyv",           # Target format
    source_format="rgb",      # Source format (default: "rgb")
)
output = transform(image)  # or transform.convert(image)
```

**Parameters:**
- `adaptor`: Target color space (str or ColorSpace enum)
- `source_format`: Source color space from data loader (str or ColorSpace enum, default: "rgb")

**Properties:**
- `adaptor`: Target adaptor name (str)
- `source_format`: Source format name (str)
- `channels`: Output channel count
- `input_channels`: Source format channel count
- `output_channels`: Channels model backbone receives

### `CameraAdaptor` (PyTorch)

```python
from edgefirst.cameraadaptor.pytorch import CameraAdaptor

adaptor = CameraAdaptor(
    adaptor="yuyv",           # Target format
    channels_last=False,      # True for NHWC input
)
output = adaptor(input_tensor)
```

**Parameters:**
- `adaptor`: Target color space (str or ColorSpace enum)
- `channels_last`: If True, input is NHWC, permuted to NCHW (default: False)

**Static Methods:**
- `compute_input_channels(args)`: Get input channels from YAML args
- `compute_output_channels(args)`: Get output channels from YAML args

### `CameraAdaptor` (TensorFlow)

```python
from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

layer = CameraAdaptor(
    adaptor="yuyv",           # Target format (None for auto-detect)
    channels_first=False,     # True for NCHW input
)
output = layer(input_tensor)
```

**Parameters:**
- `adaptor`: Target color space (str, None for auto-detect)
- `channels_first`: If True, input is NCHW, permuted to NHWC (default: False)

### `CameraAdaptorConfig`

```python
from edgefirst.cameraadaptor import CameraAdaptorConfig

config = CameraAdaptorConfig(
    adaptor="yuyv",
    input_dtype="float32",
    output_dtype="float32",
)
```

**Properties:**
- `input_channels`: Input channel count
- `output_channels`: Output channel count
- `is_quantized`: Whether config uses quantized dtypes

**Methods:**
- `to_dict()`: Convert to dictionary
- `to_metadata()`: Convert to model metadata format
- `from_dict(data)`: Create from dictionary
- `from_metadata(metadata)`: Create from model metadata

## License

Apache 2.0
