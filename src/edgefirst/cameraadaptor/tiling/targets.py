# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Target rules for one window (TILING.md §3.8)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ._types import HW, Rect
from .config import TargetRules

SUPPORTED_TARGETS = ("boxes", "classes")


@dataclass(frozen=True)
class CropResult:
    """Targets of one window after the §3.8 rules, with drop counts."""

    boxes: np.ndarray
    classes: np.ndarray
    clipped: np.ndarray
    ignore_boxes: np.ndarray
    dropped_visibility: int = 0
    dropped_min_px: int = 0
    dropped_cap: int = 0


def window_transform(
    rect: Rect, tile_hw: HW, content: Rect | None = None
) -> tuple[float, float, float, float]:
    """``(sx, sy, tx, ty)`` such that ``tile = frame * s + t``."""
    x, y, w, h = rect
    th, tw = tile_hw
    left, top, cw, ch = content if content is not None else (0, 0, tw, th)
    sx, sy = cw / w, ch / h
    return sx, sy, left - x * sx, top - y * sy


def crop_targets(
    rect: Rect,
    tile_hw: HW,
    targets: Mapping,
    rules: TargetRules,
    content: Rect | None = None,
) -> CropResult:
    """Translate, clip and filter frame-pixel targets into tile pixels."""
    reserved = sorted(set(targets) - set(SUPPORTED_TARGETS))
    if reserved:
        raise NotImplementedError(
            f"tiling targets {reserved} are reserved (TILING.md §8); only "
            f"boxes are supported"
        )
    boxes = np.asarray(targets["boxes"], dtype=np.float64).reshape(-1, 4)
    classes = np.asarray(targets["classes"], dtype=np.int32).reshape(-1)
    x, y, w, h = rect
    lo = np.array([x, y, x, y], dtype=np.float64)
    hi = np.array([x + w, y + h, x + w, y + h], dtype=np.float64)
    clip = np.clip(boxes, lo, hi)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    vis_area = np.clip(clip[:, 2] - clip[:, 0], 0, None) * np.clip(
        clip[:, 3] - clip[:, 1], 0, None
    )
    visible = vis_area > 0
    visibility = np.divide(
        vis_area, area, out=np.zeros_like(area), where=area > 0
    )
    sx, sy, tx, ty = window_transform(rect, tile_hw, content)
    tile = clip * np.array([sx, sy, sx, sy]) + np.array([tx, ty, tx, ty])
    short = np.minimum(tile[:, 2] - tile[:, 0], tile[:, 3] - tile[:, 1])
    clipped = np.any(clip != boxes, axis=1)

    dropped_vis = dropped_px = 0
    if rules.truncation == "keep":
        keep = visible
    else:
        vis_ok = visible & (visibility >= rules.min_visibility)
        px_ok = short >= rules.min_box_px
        keep = vis_ok & px_ok
        dropped_vis = int(np.sum(visible & ~vis_ok))
        dropped_px = int(np.sum(vis_ok & ~px_ok))
    if rules.truncation == "ignore":
        band = (
            visible
            & (visibility < rules.min_visibility)
            & (visibility >= rules.ignore_visibility)
        )
        ignore_boxes = tile[band].astype(np.float32)
    else:
        ignore_boxes = np.zeros((0, 4), dtype=np.float32)

    idx = np.flatnonzero(keep)
    dropped_cap = 0
    if rules.max_boxes is not None and len(idx) > rules.max_boxes:
        dropped_cap = len(idx) - rules.max_boxes
        idx = idx[: rules.max_boxes]
    return CropResult(
        tile[idx].astype(np.float32),
        classes[idx],
        clipped[idx],
        ignore_boxes,
        dropped_vis,
        dropped_px,
        dropped_cap,
    )
