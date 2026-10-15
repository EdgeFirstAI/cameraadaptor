# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `edgefirst.cameraadaptor.tiling`: the input tiling core shared by the PyTorch and TensorFlow trainers (TILING.md §3–§6), in NumPy. It provides the window sampler, epoch plan, target rules, EvenDist grid, lift and keep-best merge (bit-exact with HAL), the `tiling` metadata block, its conformance checks and sampler statistics. `frames_per_batch` sizes validation batches to a tile budget. `tiling.pixels` (window rendering) needs the `transform` extra.
- `edgefirst.cameraadaptor.pytorch.tiling`: tile rendering, per-tile targets and frame merge on tensors for tiled validation (`torch` extra).
- TILING.md: the input tiling specification and usage guide; README gains an Input Tiling section.

### Changed

- NumPy 1.25 or newer is required (was 1.21); the tiling sampler uses `Generator.spawn`.

### Fixed

- `check_torch_available`, `check_tensorflow_available`, `check_lightning_available` and `check_opencv_available` look for the package without importing it, so checking for a framework no longer loads it.
- The test suite no longer segfaults when TensorFlow and Triton are both installed. TensorFlow exports its LLVM and MLIR symbols globally, and Triton loaded afterwards binds to them and crashes, so the tests load Triton first.

## [0.1.2] - 2026-01-29

### Added

- Added GREY to `SUPPORTED_SOURCE_FORMATS` enabling grayscale-to-grayscale identity conversion
- When training with `cameraadaptor='grey'`, images can now be loaded directly as grayscale for efficiency

## [0.1.1] - 2026-01-26

### Fixed

- Corrected package name in all error messages from `edgefirst-camera-adaptor` to `edgefirst-cameraadaptor`
- Changed `[transform]` extra dependency from `opencv-python` to `opencv-python-headless` to prevent conflicts on headless servers where `opencv-python-headless` is already installed
- Fixed CHANGELOG repository URLs to point to correct GitHub path

## [0.1.0] - 2025-01-18

### Added

- Initial release
- `ColorSpace` enum with RGB, BGR, RGBA, BGRA, and YUYV support
- `CameraAdaptorTransform` for preprocessing with OpenCV
- PyTorch `CameraAdaptor` module compatible with ultralytics YAML definitions
- TensorFlow/Keras `CameraAdaptor` layer with serialization support
- PyTorch Lightning callback and mixin for experiment tracking
- `CameraAdaptorConfig` for configuration and model metadata
- Comprehensive test suite
- Documentation with quickstart guide and format references

[Unreleased]: https://github.com/EdgeFirstAI/cameraadaptor/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/EdgeFirstAI/cameraadaptor/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/EdgeFirstAI/cameraadaptor/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EdgeFirstAI/cameraadaptor/releases/tag/v0.1.0
