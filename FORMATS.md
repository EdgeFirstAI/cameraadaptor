# Color Space and Format Reference

Understanding camera formats is essential for choosing the right adaptor for your target platform.

## Format Categories

### RGB/BGR Family

Standard color formats with separate red, green, and blue channels.

| Format | Channels | Memory Layout | Description |
|--------|----------|---------------|-------------|
| RGB | 3 | Packed (RGBRGB...) | Standard RGB, common in ML frameworks |
| BGR | 3 | Packed (BGRBGR...) | Blue-Green-Red order, native to OpenCV |
| RGBA | 4 | Packed (RGBARGBA...) | RGB + Alpha transparency channel |
| BGRA | 4 | Packed (BGRABGRA...) | BGR + Alpha, common in graphics APIs |
| GREY | 1 | Single channel | Grayscale, luminance only |

**When to use:**
- **RGB**: Standard training, most datasets (ImageNet, COCO, etc.)
- **BGR**: OpenCV pipelines, i.MX PXP output, some industrial cameras
- **RGBA**: Compositing workflows, i.MX 8M Plus G2D output
- **BGRA**: Some GPU pipelines, graphics APIs
- **GREY**: Infrared cameras, thermal imaging, depth sensors, edge detection

### YUV Family

Formats that separate luminance (Y) from chrominance (U/V). More efficient for video processing.

| Format | Channels | Subsampling | Memory Layout | Description |
|--------|----------|-------------|---------------|-------------|
| YUYV | 2 | 4:2:2 | Packed (YUYVYUYV...) | Each 2 pixels share UV |
| NV12 | 1* | 4:2:0 | Semi-planar | Y plane + interleaved UV |
| NV21 | 1* | 4:2:0 | Semi-planar | Y plane + interleaved VU |

*Planar formats are treated as single-channel with special handling.

**When to use:**
- **YUYV**: USB cameras, webcams, V4L2 devices
- **NV12**: Hardware codecs, Jetson cameras, i.MX cameras
- **NV21**: Android camera API

#### Why YUV?

1. **Human vision**: We're more sensitive to brightness than color detail
2. **Compression**: Chrominance can be subsampled without perceived quality loss
3. **Hardware**: Many cameras and codecs work natively in YUV

#### Subsampling Patterns

YUV subsampling is expressed as J:a:b notation:

| Pattern | Meaning | Data Reduction |
|---------|---------|----------------|
| 4:4:4 | Full resolution for Y, U, and V | None |
| 4:2:2 | U and V at half horizontal resolution | 33% vs RGB |
| 4:2:0 | U and V at half horizontal and vertical | 50% vs RGB |

### Bayer Family (Roadmap)

Raw sensor formats before demosaicing. Each pixel captures only one color.

| Format | Pattern | Description |
|--------|---------|-------------|
| RGGB | R at (0,0) | Common Bayer pattern |
| BGGR | B at (0,0) | Alternative pattern |
| GRBG | G at (0,0), R at (0,1) | Rotated pattern |
| GBRG | G at (0,0), B at (0,1) | Rotated pattern |

**When to use:**
- Platforms without ISP (image signal processor)
- Maximum flexibility in image processing
- Raw camera sensors

## Memory Layout Details

### Packed vs Planar

**Packed** (YUYV, RGB, BGR):
```
Pixel 0    Pixel 1    Pixel 2    ...
[Y0 U0]    [Y1 V0]    [Y2 U1]    ...
```

**Semi-Planar** (NV12):
```
Y Plane:  [Y0] [Y1] [Y2] [Y3] ...
UV Plane: [U0 V0] [U1 V1] ...
```

### Channel Order

```
RGB:  [R, G, B]
BGR:  [B, G, R]
RGBA: [R, G, B, A]
BGRA: [B, G, R, A]
YUYV: [Y, U/V] (alternating U and V per pixel pair)
```

## YUYV Format Details

YUYV (also known as YUY2) is a packed YUV 4:2:2 format commonly used by USB cameras.

### Memory Layout

For every 2 pixels, 4 bytes are used:

```
Byte:  [Y0] [U0] [Y1] [V0] [Y2] [U1] [Y3] [V1] ...
Pixel:  ├─ P0 ─┤  ├─ P1 ─┤  ├─ P2 ─┤  ├─ P3 ─┤
```

- Each pixel has its own Y (luminance)
- Adjacent pixel pairs share U and V (chrominance)

### In CameraAdaptor

When processed as 2-channel tensor:
- **Channel 0**: Y (luminance) - full resolution, one value per pixel
- **Channel 1**: Interleaved UV (chrominance) - shared between pixel pairs

```python
# YUYV produces 2-channel output
transform = CameraAdaptorTransform("yuyv")
yuyv = transform(rgb_image)  # Shape: (H, W, 2)

# In model (PyTorch NCHW)
adaptor = CameraAdaptor("yuyv")
x = torch.randn(1, 2, 224, 224)  # 2 channels
```

### Common Use Cases

- USB Video Class (UVC) cameras
- V4L2 capture devices
- Consumer webcams

### FourCC Codes

Related formats and their FourCC codes:
- **YUYV** / **YUY2**: Packed 4:2:2, YUYV order
- **UYVY**: Packed 4:2:2, UYVY order (U first)
- **YV16**: Planar 4:2:2 (Y plane, V plane, U plane)

## NV12 Format Details (Roadmap)

### Memory Layout

Data is organized in two planes:

```
Y Plane (full resolution):
[Y00] [Y01] [Y02] [Y03]
[Y10] [Y11] [Y12] [Y13]
[Y20] [Y21] [Y22] [Y23]
[Y30] [Y31] [Y32] [Y33]

UV Plane (half resolution, interleaved):
[U00 V00] [U01 V01]
[U10 V10] [U11 V11]
```

Each 2x2 block of Y pixels shares one U and one V value.

### Common Use Cases

- NVIDIA Jetson cameras
- NXP i.MX cameras
- Hardware video codecs
- V4L2 DMA buffers

## Bayer Pattern Details (Roadmap)

A typical sensor uses a 2x2 repeating pattern:

```
RGGB Pattern:
┌───┬───┐
│ R │ G │
├───┼───┤
│ G │ B │
└───┴───┘
```

The "G" appears twice because human vision is most sensitive to green.

### Pattern Variants

| Pattern | Layout | First Pixel |
|---------|--------|-------------|
| RGGB | R-G / G-B | Red |
| BGGR | B-G / G-R | Blue |
| GRBG | G-R / B-G | Green (R to right) |
| GBRG | G-B / R-G | Green (B to right) |

### Why Use Raw Bayer?

**Traditional Pipeline (with ISP):**
```
Sensor → ISP Demosaic → RGB → Model
           (hardware)
```

**Without ISP:**
```
Sensor → Raw Bayer → Model (trained on Bayer)
```

**Benefits:**
- No ISP hardware required
- Lower latency
- Full control over processing
- Access to raw sensor data

**Challenges:**
- Model must learn color interpolation
- Requires Bayer-format training data
- Different sensor patterns need different models

## Channel Ordering in Models

### Planar vs Packed: Same Concept, Different Terminology

The ML framework terminology (`channels_first`/`channels_last`) maps directly to
video/camera terminology (planar/packed):

| ML Framework Term | Video/Camera Term | Memory Layout | Example |
|-------------------|-------------------|---------------|---------|
| **channels_first** (NCHW) | **Planar** | All R, then all G, then all B | `[RRRR...][GGGG...][BBBB...]` |
| **channels_last** (NHWC) | **Packed/Interleaved** | RGB values interleaved per pixel | `[RGB][RGB][RGB]...` |

**Memory layout comparison for a 2x2 RGB image:**

```
Planar / channels_first (NCHW):
R plane: [R00 R01 R10 R11]
G plane: [G00 G01 G10 G11]
B plane: [B00 B01 B10 B11]
Memory: R00 R01 R10 R11 G00 G01 G10 G11 B00 B01 B10 B11

Packed / channels_last (NHWC):
Memory: R00 G00 B00 R01 G01 B01 R10 G10 B10 R11 G11 B11
```

### YUV Planar Formats (NV12, I420)

YUV planar formats like NV12 store the Y (luminance) plane separately from the
UV (chrominance) data:

```
NV12 (semi-planar YUV 4:2:0):
├── Y Plane:  [Y Y Y Y Y Y Y Y ...]   (full resolution)
└── UV Plane: [U V U V U V ...]       (half resolution, interleaved)

I420 (fully planar YUV 4:2:0):
├── Y Plane: [Y Y Y Y ...]
├── U Plane: [U U ...]
└── V Plane: [V V ...]
```

These are inherently planar (channels_first) formats. When used with TensorFlow
(which defaults to channels_last/NHWC), a transpose operation may be needed.

### Framework Defaults

### Channels-First vs Channels-Last

Deep learning frameworks differ in their default tensor layouts:

| Framework | Default Layout | Dimension Order |
|-----------|---------------|--------------------|
| PyTorch | Channels-First | NCHW (batch, channel, height, width) |
| TensorFlow/Keras | Channels-Last | NHWC (batch, height, width, channel) |
| ONNX | Channels-First | NCHW |
| TFLite | Channels-Last | NHWC |

### CameraAdaptor Channel Handling

**PyTorch** (`channels_last=False` default):
```python
# Default: expects NCHW input
adaptor = CameraAdaptor("yuyv")
x = torch.randn(1, 2, 224, 224)  # NCHW
y = adaptor(x)  # Output: NCHW

# With channels_last=True: accepts NHWC, outputs NCHW
adaptor = CameraAdaptor("yuyv", channels_last=True)
x = torch.randn(1, 224, 224, 2)  # NHWC from camera
y = adaptor(x)  # Output: NCHW for PyTorch backbone
```

**TensorFlow** (`channels_first=False` default):
```python
# Default: expects NHWC input
layer = CameraAdaptor("yuyv")
x = tf.random.normal((1, 224, 224, 2))  # NHWC
y = layer(x)  # Output: NHWC

# With channels_first=True: accepts NCHW, outputs NHWC
layer = CameraAdaptor("yuyv", channels_first=True)
x = tf.random.normal((1, 2, 224, 224))  # NCHW
y = layer(x)  # Output: NHWC for TensorFlow backbone
```

## YUV to/from RGB Conversion

The conversion matrices (BT.601):

**YUV to RGB:**
```
R = Y + 1.402 * (V - 128)
G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)
B = Y + 1.772 * (U - 128)
```

**RGB to YUV:**
```
Y = 0.299 * R + 0.587 * G + 0.114 * B
U = -0.169 * R - 0.331 * G + 0.500 * B + 128
V = 0.500 * R - 0.419 * G - 0.081 * B + 128
```

## Training Considerations for Non-RGB Formats

When training models with formats other than RGB, several factors can affect accuracy compared to RGB-trained models. Users should evaluate results on their specific dataset and model to ensure requirements are met.

### Pre-trained Weights

Most pre-trained weights (ImageNet, COCO, etc.) are trained on RGB images with 3 channels:

| Format | Channels | Pre-trained Weight Compatibility |
|--------|----------|----------------------------------|
| RGB | 3 | Full compatibility |
| BGR | 3 | Requires R-B weight swap (not provided by CameraAdaptor) |
| RGBA | 4 | Partial (alpha channel ignored, RGB weights usable) |
| BGRA | 4 | Partial (alpha ignored, R-B weight swap needed) |
| YUYV | 2 | Incompatible (different channel count and content) |
| GREY | 1 | Incompatible (different channel count and content) |

For BGR/BGRA formats, the first convolutional layer weights would need R-B channel swapping to match the input order. CameraAdaptor does not currently provide weight conversion utilities.

For formats with different channel counts and semantics (YUYV, GREY), pre-trained RGB weights are not directly applicable because:
- **Channel count differs**: YUYV has 2 channels (Y + UV), GREY has 1 channel
- **Channel content differs**: Y (luma) and UV (chroma) have different semantics than R, G, B color channels

These formats may require:
- Random initialization of the first layer (loses transfer learning benefit)
- Weight adaptation strategies (e.g., deriving Y weights from RGB luminance formula)
- Additional training epochs to compensate

### Normalization

Standard training normalization divides pixel values by 255 to produce a 0-1 range. This approach works consistently across all formats for quantized deployment:

```python
# Training: normalize to 0-1
image = image.float() / 255.0

# Quantized inference: raw uint8 (0-255), model calibrated for this range
```

While different formats have different channel semantics (e.g., YUYV has luma and chroma channels with different value distributions), using uniform `/255` normalization maintains compatibility with quantization workflows where the model is calibrated to accept raw uint8 input.

### Quantization

For quantized (INT8) deployment:

1. **Training** uses float32 with `/255` normalization
2. **Calibration** uses representative uint8 data in the target format
3. **Inference** uses raw uint8 input directly (no normalization)

The model learns to handle the normalized range during training, and calibration ensures the quantized model correctly processes the uint8 input range.

### Expected Accuracy Variation

Models trained on non-RGB formats may show slightly different accuracy compared to RGB:

| Factor | Typical Impact | Notes |
|--------|----------------|-------|
| BGR vs RGB | Minimal (<0.1%) | Simple channel swap, weights transfer well |
| RGBA/BGRA vs RGB | Minimal (<0.1%) | Alpha dropped, RGB/BGR weights transfer |
| YUYV vs RGB | Small (1-3%) | Channel mismatch, chroma subsampling |
| GREY vs RGB | Variable | Depends on task's color sensitivity |

The actual impact depends on:
- Specific dataset characteristics
- Model architecture
- Training hyperparameters
- Whether the task relies on color information

**Recommendation:** Always compare validation metrics between RGB and target format training on your specific dataset to ensure the results meet your accuracy requirements.

### YUV 4:2:2 Chroma Subsampling

YUYV format subsamples chrominance (color) information by half horizontally. This is inherent to the format and cannot be avoided:

- Each pixel has full luma (Y) resolution
- Adjacent pixel pairs share chroma (U/V) values
- Fine color detail between horizontal pixels is lost

For tasks that rely heavily on subtle color differences, this may contribute to reduced accuracy compared to RGB.

## Format Support Status

| Format | Transform | PyTorch | TensorFlow | Status |
|--------|-----------|---------|------------|--------|
| RGB | ✅ | ✅ | ✅ | Supported |
| BGR | ✅ | ✅ | ✅ | Supported |
| RGBA | ✅ | ✅ | ✅ | Supported |
| BGRA | ✅ | ✅ | ✅ | Supported |
| GREY | ✅ | ✅ | ✅ | Supported |
| YUYV | ✅ | ✅ | ✅ | Supported |
| NV12 | 🔜 | 🔜 | 🔜 | Roadmap |
| NV21 | 🔜 | 🔜 | 🔜 | Roadmap |
| RGGB | 🔜 | 🔜 | 🔜 | Roadmap |
| BGGR | 🔜 | 🔜 | 🔜 | Roadmap |
| GRBG | 🔜 | 🔜 | 🔜 | Roadmap |
| GBRG | 🔜 | 🔜 | 🔜 | Roadmap |
