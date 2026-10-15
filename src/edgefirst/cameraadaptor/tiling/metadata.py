# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""The v1 ``tiling`` section of a model's edgefirst.json (usermanual
metadata.md, Tiling).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from .config import SPEC_VERSION, TilingConfig

TILING_MAX_DET = 300
SAMPLER = "edgefirst-tiling"
MODES = ("tiled", "whole_frame")


def _one_of(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(
            f"tiling metadata {name} must be one of {allowed}, got {value!r}"
        )
    return value


def training_record(cfg: TilingConfig, session: str) -> dict:
    """How the weights were trained; written once by the training session."""
    return {
        "session": session,
        "sampler": SAMPLER,
        "version": SPEC_VERSION,
        "params": cfg.to_dict(),
    }


def export_record(
    session: str,
    *,
    mode: str,
    input_hw: Sequence[int],
    full_frame: bool,
    weights_from: str | None = None,
) -> dict:
    """How this model was exported, including the calibration geometry it
    expects.
    """
    record: dict = {"session": session}
    if weights_from is not None and weights_from != session:
        record["weights_from"] = weights_from
    record["calibration"] = {
        "resize": "tiled"
        if _one_of("mode", mode, MODES) == "tiled"
        else "letterbox",
        "input_shape": [int(input_hw[0]), int(input_hw[1])],
        "full_frame": bool(full_frame),
    }
    return record


def build_tiling_block(
    cfg: TilingConfig,
    *,
    session: str,
    per_tile_nms: str = "class_aware",
    full_frame: bool = False,
    per_tile_iou: float = 0.5,
    per_tile_score: float = 0.001,
    merge_mode: str = "keep_best",
    merge_metric: str = "ios",
    merge_threshold: float = 0.5,
    class_agnostic: bool = False,
    max_det: int = TILING_MAX_DET,
    own_above_area: float = 0.02,
) -> dict:
    """Section for a model exported by the session that trained it: tiled at
    the training tile.
    """
    tile = [int(cfg.tile_size[0]), int(cfg.tile_size[1])]
    return {
        "version": 1,
        "mode": "tiled",
        "tile": tile,
        "grid": {
            "algorithm": "evendist",
            "version": 1,
            "min_overlap": cfg.grid_min_overlap,
        },
        "fit": "letterbox",
        "pad": [114, 114, 114],
        "per_tile": {
            "nms": _one_of(
                "per_tile.nms",
                per_tile_nms,
                ("class_aware", "class_agnostic", "none"),
            ),
            "iou": per_tile_iou,
            "score": per_tile_score,
        },
        "merge": {
            "mode": _one_of("merge.mode", merge_mode, ("keep_best", "union")),
            "metric": _one_of("merge.metric", merge_metric, ("ios", "iou")),
            "threshold": merge_threshold,
            "class_agnostic": class_agnostic,
            "max_det": max_det,
        },
        "full_frame": {
            "enabled": bool(full_frame),
            "own_above_area": own_above_area,
        },
        "training": training_record(cfg, session),
        "export": export_record(
            session, mode="tiled", input_hw=tile, full_frame=full_frame
        ),
    }


def reexport_tiling_block(
    source: Mapping,
    *,
    mode: str,
    input_hw: Sequence[int],
    export_session: str,
    weights_from: str,
) -> dict:
    """Section for weights re-exported by another session; ``training`` is
    copied unchanged.

    In ``tiled`` mode the tile is the new input; in ``whole_frame`` mode the
    tile and the grid settings stay as trained and the whole-frame pass is
    disabled.
    """
    block = copy.deepcopy(dict(source))
    block["mode"] = _one_of("mode", mode, MODES)
    if mode == "tiled":
        block["tile"] = [int(input_hw[0]), int(input_hw[1])]
    full_frame = dict(block.get("full_frame") or {})
    full_frame["enabled"] = (
        bool(full_frame.get("enabled", False)) and mode == "tiled"
    )
    block["full_frame"] = full_frame
    block["export"] = export_record(
        export_session,
        mode=mode,
        input_hw=input_hw,
        full_frame=full_frame["enabled"],
        weights_from=weights_from,
    )
    return block
