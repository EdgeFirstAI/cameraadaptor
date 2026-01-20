# Platform-Specific Guidance

EdgeFirst CameraAdaptor enables training models for the native formats output by target platform hardware accelerators. This guide covers platform-specific recommendations.

## NXP i.MX Platforms

### Platform Overview

| SoC | NPU | 2D Accelerator | Recommended Format | Notes |
|-----|-----|----------------|-------------------|-------|
| i.MX 93 | Arm Ethos-U65 | PXP | **BGR** | PXP outputs BGR |
| i.MX 8M Plus | NPU (2.3 TOPS) | G2D | **RGBA** | G2D outputs RGBA |
| i.MX 95 | eIQ Neutron | ISI/ISP | NV12, YUYV | ISI provides format flexibility |

### i.MX 93 with PXP

The i.MX 93 includes the Pixel Processing Pipeline (PXP) for 2D graphics operations like scaling and color space conversion. The PXP outputs in **BGR format**.

> [!IMPORTANT]
> Train models with BGR format when targeting i.MX 93 with PXP preprocessing.

#### Training Configuration

```python
from edgefirst.cameraadaptor import CameraAdaptorTransform
from edgefirst.cameraadaptor.pytorch import CameraAdaptor
import torch.nn as nn

# Transform for training data
transform = CameraAdaptorTransform("bgr")

class IMX93Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adaptor = CameraAdaptor("bgr")
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 3 channels for BGR
            # ... rest of model
        )

    def forward(self, x):
        x = self.adaptor(x)
        return self.backbone(x)
```

#### Inference Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Camera  │───>│  PXP    │───>│ Memory  │───>│ Ethos-U │
│         │    │ (scale) │    │ (BGR)   │    │  NPU    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                   │
                   └── Outputs BGR directly
```

### i.MX 8M Plus with G2D

The i.MX 8M Plus includes the Graphics 2D (G2D) engine for hardware-accelerated 2D operations. G2D supports scaling, rotation, and color space conversion, outputting in **RGBA format**.

> [!IMPORTANT]
> Train models with RGBA format when targeting i.MX 8M Plus with G2D preprocessing. The CameraAdaptor automatically slices off the alpha channel.

#### Training Configuration

```python
from edgefirst.cameraadaptor import CameraAdaptorTransform
from edgefirst.cameraadaptor.pytorch import CameraAdaptor
import torch.nn as nn

# Transform for training data
transform = CameraAdaptorTransform("rgba")

class IMX8MPlusModel(nn.Module):
    def __init__(self):
        super().__init__()
        # RGBA input (4 channels) -> RGB output (3 channels)
        self.adaptor = CameraAdaptor("rgba")
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 3 channels after alpha sliced
            # ... rest of model
        )

    def forward(self, x):
        x = self.adaptor(x)  # (N, 4, H, W) -> (N, 3, H, W)
        return self.backbone(x)
```

#### TensorFlow/Keras Configuration

```python
import tensorflow as tf
from edgefirst.cameraadaptor.tensorflow import CameraAdaptor

# Model receives RGBA from G2D
inputs = tf.keras.Input(shape=(224, 224, 4))  # RGBA input
x = CameraAdaptor("rgba")(inputs)  # Drops alpha -> 3 channels
x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
# ... rest of model
```

#### Inference Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Camera  │───>│  G2D    │───>│ Memory  │───>│  NPU    │
│         │    │ (scale) │    │ (RGBA)  │    │(TFLite) │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                   │
                   └── Outputs RGBA, model expects RGBA
                       and slices alpha internally
```

### i.MX 95 with ISI/ISP

The i.MX 95 features the Image Sensing Interface (ISI) and Image Signal Processor (ISP) which provide more flexibility in output format selection.

#### Common ISI Output Formats

| Format | Channels | Use Case |
|--------|----------|----------|
| NV12 | Semi-planar | Hardware codec integration |
| YUYV | 2 | Direct camera capture |
| RGB/BGR | 3 | Standard vision models |

#### Training for ISI Pipeline

```python
from edgefirst.cameraadaptor import CameraAdaptorTransform
from edgefirst.cameraadaptor.pytorch import CameraAdaptor

# Choose format based on ISI configuration
# NV12 for hardware codec path
transform = CameraAdaptorTransform("nv12")  # Roadmap

# Or YUYV for direct capture
transform = CameraAdaptorTransform("yuyv")

class IMX95Model(nn.Module):
    def __init__(self, adaptor="yuyv"):
        super().__init__()
        self.adaptor = CameraAdaptor(adaptor)
        in_channels = CameraAdaptor.compute_output_channels(adaptor)
        self.backbone = create_backbone(in_channels=in_channels)
```

## On-Target Deployment with EdgeFirst HAL

[EdgeFirst HAL](https://github.com/EdgeFirstAI/hal) provides optimized preprocessing pipelines for on-target inference and benchmarking.

### Integration Example

```python
# Training with CameraAdaptor
from edgefirst.cameraadaptor import CameraAdaptorConfig

config = CameraAdaptorConfig(
    adaptor="bgr",
    input_dtype="uint8",
    output_dtype="uint8",
)

# Embed config in exported model for HAL to read
metadata = config.to_metadata()
```

HAL reads the embedded configuration and automatically configures the preprocessing pipeline to match training expectations.

Refer to the [EdgeFirst HAL documentation](https://github.com/EdgeFirstAI/hal) for benchmarking and deployment guidance.

## Performance Considerations

### Memory Bandwidth

| Configuration | Memory Operations | Bandwidth Impact |
|--------------|-------------------|------------------|
| Camera → RGB Convert → NPU | 2 reads + 2 writes | High |
| Camera → NPU (native format) | 1 read + 1 write | Low |

Training for native formats eliminates conversion, reducing memory bandwidth by up to 50%.

### DMA Buffer Zero-Copy

For optimal performance on i.MX platforms, use DMA buffers to share memory between camera capture and NPU inference:

```c
// V4L2 setup with DMABUF
struct v4l2_requestbuffers req;
req.count = 4;
req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
req.memory = V4L2_MEMORY_DMABUF;
ioctl(fd, VIDIOC_REQBUFS, &req);
```

## Troubleshooting

### Model expects wrong number of channels

**Symptom**: Shape mismatch error during inference

**Solution**: Verify the adaptor setting matches between training and inference:

```python
# Check expected channels
from edgefirst.cameraadaptor.pytorch import CameraAdaptor

print(CameraAdaptor.compute_input_channels("rgba"))   # 4 (model input)
print(CameraAdaptor.compute_output_channels("rgba"))  # 3 (backbone input)
```

### Colors appear incorrect

**Symptom**: Model predictions are poor despite good training accuracy

**Cause**: Format mismatch between training transform and target hardware

**Solution**: Verify your training format matches the target platform:
- i.MX 93 with PXP → use `bgr`
- i.MX 8M Plus with G2D → use `rgba`

### Quantization issues

**Symptom**: Accuracy drops significantly after TFLite conversion

**Solution**: Use representative dataset with correct format during quantization:

```python
from edgefirst.cameraadaptor import CameraAdaptorTransform

transform = CameraAdaptorTransform("bgr")

def representative_dataset():
    for image in calibration_images:
        bgr = transform(image)
        yield [bgr[np.newaxis, ...].astype(np.uint8)]
```

## Future Platform Support

Additional platforms will be documented as support is added:

- NVIDIA Jetson (Orin, Xavier, Nano)
- Qualcomm platforms
- Other edge AI accelerators

See [EdgeFirst HAL](https://github.com/EdgeFirstAI/hal) for the latest platform support.
