# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Fixed-shape window packing for graph frameworks (TILING.md §7.4).

A window's geometry and targets are variable-length records; a tf.data or XLA
pipeline needs static shapes. These pack them into fixed arrays without
reordering or dropping anything: the box cap is the §3.8 ``targets.max_boxes``
rule, applied by ``crop_targets`` before packing.
"""

from __future__ import annotations

import numpy as np

from .grid import TilePlacement
from .sampler import Window
from .targets import CropResult

GEOMETRY_FIELDS = ("x", "y", "w", "h", "left", "top", "cw", "ch")
"""Order of ``pack_geometry``'s eight values."""

PAD = -1.0
"""Value of every field of an unused target row."""


def pack_geometry(window: Window) -> np.ndarray:
    """``int32[8]``: the frame rectangle ``(x, y, w, h)`` and the content
    placement on the tile ``(left, top, cw, ch)``, which is the whole tile
    when the window needs no letterbox.
    """
    th, tw = window.out_hw
    content = window.content if window.content is not None else (0, 0, tw, th)
    return np.asarray((*window.rect, *content), dtype=np.int32)


def pack_placements(placements: list[TilePlacement]) -> np.ndarray:
    """``int32[n, 8]`` geometry of grid tiles, in ``pack_geometry``'s layout,
    so validation renders tiles with the same op as training windows.
    """
    rows = []
    for p in placements:
        th, tw = p.tile_hw
        content = p.content if p.content is not None else (0, 0, tw, th)
        rows.append((*p.rect, *content))
    return np.asarray(rows, dtype=np.int32).reshape(-1, 8)


def pack_targets(
    targets: CropResult | Window, tile_hw: tuple[int, int], max_boxes: int
) -> np.ndarray:
    """``float32[max_boxes, 5]`` rows of normalised ``(cx, cy, w, h, class)``
    in input order, unused rows filled with ``PAD``.

    Raises ValueError when there are more boxes than rows: set
    ``targets.max_boxes`` to the packing size so the cap is the §3.8 rule and
    is counted, rather than a silent truncation here.
    """
    result = targets.targets if isinstance(targets, Window) else targets
    boxes, classes = result.boxes, result.classes
    if len(boxes) > max_boxes:
        raise ValueError(
            f"{len(boxes)} boxes do not fit {max_boxes} rows; set "
            f"targets.max_boxes to {max_boxes}"
        )
    th, tw = tile_hw
    out = np.full((max_boxes, 5), PAD, dtype=np.float32)
    if len(boxes):
        b = np.asarray(boxes, dtype=np.float32)
        out[: len(b), 0] = (b[:, 0] + b[:, 2]) / 2 / tw
        out[: len(b), 1] = (b[:, 1] + b[:, 3]) / 2 / th
        out[: len(b), 2] = (b[:, 2] - b[:, 0]) / tw
        out[: len(b), 3] = (b[:, 3] - b[:, 1]) / th
        out[: len(b), 4] = classes
    return out


__all__ = [
    "GEOMETRY_FIELDS",
    "PAD",
    "pack_geometry",
    "pack_placements",
    "pack_targets",
]
