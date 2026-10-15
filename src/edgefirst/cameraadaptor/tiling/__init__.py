# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""EdgeFirst input tiling (TILING.md §3–§6).

Window sampler, epoch plan, target rules, EvenDist grid, lift and merge
reference and the runtime metadata block, in NumPy, shared by the PyTorch and
TensorFlow trainers. ``pixels`` needs the ``transform`` extra (OpenCV) and
``edgefirst.cameraadaptor.pytorch.tiling`` the ``torch`` extra.
"""

from .config import (
    KINDS,
    SPEC_VERSION,
    Mix,
    ScaleConfig,
    TargetRules,
    TilingConfig,
)
from .contract import (
    ContractError,
    UnsupportedVersion,
    check_converter_tiling,
    check_snapshot_geometry,
    check_snapshot_params,
    model_input_hw,
    parse_session_id,
    validate_tiling_block,
)
from .grid import (
    TilePlacement,
    axis_origins,
    fit_content,
    frames_per_batch,
    letterbox_rect,
    tile_grid,
)
from .merge import lift_tile_boxes, merge_tiled_detections
from .metadata import (
    SAMPLER,
    TILING_MAX_DET,
    build_tiling_block,
    export_record,
    reexport_tiling_block,
    training_record,
)
from .plan import EpochPlan, PlannedTile, batch_layout, gt_extent, plan_epoch
from .sampler import Window, make_rng, repeat_factors, sample_windows
from .stats import SamplerStats
from .targets import CropResult, crop_targets, window_transform

__all__ = [
    "KINDS",
    "SPEC_VERSION",
    "Mix",
    "ScaleConfig",
    "TargetRules",
    "TilingConfig",
    "TilePlacement",
    "axis_origins",
    "fit_content",
    "frames_per_batch",
    "letterbox_rect",
    "tile_grid",
    "lift_tile_boxes",
    "merge_tiled_detections",
    "SAMPLER",
    "TILING_MAX_DET",
    "build_tiling_block",
    "export_record",
    "reexport_tiling_block",
    "training_record",
    "CropResult",
    "crop_targets",
    "window_transform",
    "Window",
    "make_rng",
    "repeat_factors",
    "sample_windows",
    "SamplerStats",
    "EpochPlan",
    "PlannedTile",
    "batch_layout",
    "gt_extent",
    "plan_epoch",
    "ContractError",
    "UnsupportedVersion",
    "check_converter_tiling",
    "check_snapshot_geometry",
    "check_snapshot_params",
    "model_input_hw",
    "parse_session_id",
    "validate_tiling_block",
]
