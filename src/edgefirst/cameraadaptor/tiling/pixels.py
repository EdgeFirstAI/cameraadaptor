# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Pixel stage for a sampled window (TILING.md §3.6, §3.7)."""

from __future__ import annotations

import cv2
import numpy as np


def render_window(
    frame: np.ndarray,
    rect,
    tile_hw,
    content=None,
    resize: str = "antialias",
    pad: int = 114,
) -> np.ndarray:
    """Crop ``rect`` and fit it to the tile.

    Native windows are pure crops. Downscales use area filtering (antialiased)
    unless ``resize == "bilinear"``; upscales are bilinear. ``content`` places
    a letterboxed crop on a ``pad``-filled tile.
    """
    x, y, w, h = rect
    th, tw = tile_hw
    crop = frame[y : y + h, x : x + w]
    left, top, cw, ch = content if content is not None else (0, 0, tw, th)
    if (cw, ch) != (w, h):
        downscale = cw < w or ch < h
        interp = (
            cv2.INTER_AREA
            if downscale and resize == "antialias"
            else cv2.INTER_LINEAR
        )
        crop = cv2.resize(crop, (cw, ch), interpolation=interp)
        if crop.ndim == 2:
            crop = crop[..., None]
    if content is None:
        return np.ascontiguousarray(crop)
    tile = np.full((th, tw, frame.shape[2]), pad, dtype=frame.dtype)
    tile[top : top + ch, left : left + cw] = crop
    return tile
