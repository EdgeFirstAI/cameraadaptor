# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""EvenDist v1 grid and letterbox placement (TILING.md §6.3), bit-exact with
HAL.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def _round_half_away(value: float) -> int:
    """Rust ``round`` for non-negative values: half away from zero."""
    return int(math.floor(value + 0.5))


def axis_origins(length: int, tile: int, min_overlap: float) -> list[int]:
    """EvenDist origins along one axis, matching HAL ``axis_origins``.

    ``min_overlap`` is stored as float32 and widened to float64; each origin is
    computed in float32 and rounded half away from zero.
    """
    if length <= tile:
        return [0]
    last = length - tile
    overlap = float(np.float32(min_overlap))
    max_step = max(1, math.floor((1.0 - overlap) * tile))
    n = -(-last // max_step)
    f_last, f_n = np.float32(last), np.float32(n)
    return [
        _round_half_away(float(np.float32(i) * f_last / f_n))
        for i in range(n + 1)
    ]


def letterbox_rect(
    src_w: int, src_h: int, dst_w: int, dst_h: int
) -> tuple[int, int, int, int]:
    """Centred, aspect-preserving placement ``(left, top, width, height)``, as
    HAL ``letterbox_rect``.
    """
    if src_w == 0 or src_h == 0:
        return 0, 0, dst_w, dst_h
    src_aspect = src_w / src_h
    if src_aspect > dst_w / dst_h:
        new_w, new_h = dst_w, max(1, _round_half_away(dst_w / src_aspect))
    else:
        new_w, new_h = max(1, _round_half_away(dst_h * src_aspect)), dst_h
    return max(dst_w - new_w, 0) // 2, max(dst_h - new_h, 0) // 2, new_w, new_h


def fit_content(
    crop_w: int, crop_h: int, tile_hw: tuple[int, int], fit: str = "letterbox"
) -> tuple[int, int, int, int] | None:
    """Content rectangle of a crop on the tile, or ``None`` when it fills the
    tile.
    """
    th, tw = tile_hw
    if fit == "stretch":
        return None
    content = letterbox_rect(crop_w, crop_h, tw, th)
    return None if content == (0, 0, tw, th) else content


@dataclass(frozen=True)
class TilePlacement:
    """One grid tile: native crop in frame pixels and its placement on the
    tile.
    """

    index: int
    origin: tuple[int, int]
    crop_size: tuple[int, int]
    tile_hw: tuple[int, int]
    content: tuple[int, int, int, int] | None

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """The crop as ``(x, y, w, h)`` in frame pixels."""
        return (*self.origin, *self.crop_size)

    @property
    def letterbox(self) -> tuple[float, float, float, float] | None:
        """Normalised content bounds on the model input (HAL
        ``TilePlacement.letterbox``).
        """
        if self.content is None:
            return None
        th, tw = np.float32(self.tile_hw[0]), np.float32(self.tile_hw[1])
        left, top, w, h = (np.float32(v) for v in self.content)
        return (
            float(left / tw),
            float(top / th),
            float((left + w) / tw),
            float((top + h) / th),
        )


def tile_grid(
    frame_hw: tuple[int, int],
    tile_hw: tuple[int, int],
    min_overlap: float,
    fit: str = "letterbox",
) -> list[TilePlacement]:
    """Row-major EvenDist grid. An axis shorter than the tile gets one
    whole-axis crop.
    """
    fh, fw = int(frame_hw[0]), int(frame_hw[1])
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    crop = (min(tw, fw), min(th, fh))
    content = fit_content(crop[0], crop[1], (th, tw), fit)
    origins = [
        (x, y)
        for y in axis_origins(fh, th, min_overlap)
        for x in axis_origins(fw, tw, min_overlap)
    ]
    return [
        TilePlacement(i, o, crop, (th, tw), content)
        for i, o in enumerate(origins)
    ]


def frames_per_batch(
    frame_shapes: Iterable[Sequence[int]],
    tile_hw: tuple[int, int],
    min_overlap: float,
    tile_budget: int,
) -> int:
    """Frames per validation batch so that a batch's tiles fit ``tile_budget``
    (TILING.md §5).

    Sized by the frame shape with the most grid tiles, so mixed-resolution
    splits never exceed the budget.
    """
    shapes = {(int(h), int(w)) for h, w in frame_shapes}
    most = max(
        (len(tile_grid(hw, tile_hw, min_overlap)) for hw in shapes), default=1
    )
    return max(1, int(tile_budget) // most)
