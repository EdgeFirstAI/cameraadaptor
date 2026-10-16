# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Sampler statistics (TILING.md §3.10) as a flat float64 vector.

The flat layout lets data-loader workers accumulate into shared memory; the
trainer turns the vector into a per-epoch summary.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ._types import ArrayLike

if TYPE_CHECKING:
    from .sampler import Window

_SCALARS = (
    "windows",
    "positive",
    "background",
    "full_frame",
    "background_fallback",
    "empty",
    "native",
    "boxes",
    "clipped",
    "dropped_visibility",
    "dropped_min_px",
    "dropped_cap",
    "decodes",
    "samples",
    "positive_boxes",
)
ANCHOR_BINS = 8
OBJECT_EDGES = (0, 1, 2, 3, 4, 8, 16)
SCALE_EDGES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)

_ANCHOR = len(_SCALARS)
_OBJECTS = _ANCHOR + ANCHOR_BINS * ANCHOR_BINS
_SCALE = _OBJECTS + len(OBJECT_EDGES)


class SamplerStats:
    """Per-window sampler statistics (§3.10) in one flat float64 vector."""

    SIZE = _SCALE + len(SCALE_EDGES) - 1

    def __init__(self, vector: ArrayLike | None = None):
        self.v = (
            np.zeros(self.SIZE, np.float64)
            if vector is None
            else np.array(vector, np.float64)
        )

    @staticmethod
    def index(key: str) -> int:
        """Position of a scalar counter in the vector."""
        return _SCALARS.index(key)

    def add(self, key: str, n: float = 1) -> None:
        """Add ``n`` to a scalar counter."""
        self.v[self.index(key)] += n

    def __iadd__(self, other: SamplerStats) -> SamplerStats:
        self.v += other.v
        return self

    def record(self, window: Window) -> None:
        """Count one sampled window."""
        self.add("windows")
        self.add(window.kind)
        if window.fallback:
            self.add("background_fallback")
        n = len(window.boxes)
        if n == 0:
            self.add("empty")
        self.add("boxes", n)
        if window.kind == "positive":
            self.add("positive_boxes", n)
        self.add("clipped", int(np.sum(window.targets.clipped)))
        self.add("dropped_visibility", window.targets.dropped_visibility)
        self.add("dropped_min_px", window.targets.dropped_min_px)
        self.add("dropped_cap", window.targets.dropped_cap)
        self.v[
            _OBJECTS + int(np.searchsorted(OBJECT_EDGES, n, side="right")) - 1
        ] += 1
        if window.kind != "full_frame":
            if window.rect[2] == window.out_hw[1]:
                self.add("native")
            else:
                log_inv = math.log(window.out_hw[1] / window.rect[2])
                i = int(
                    np.clip(
                        np.searchsorted(SCALE_EDGES, log_inv, side="right")
                        - 1,
                        0,
                        len(SCALE_EDGES) - 2,
                    )
                )
                self.v[_SCALE + i] += 1
        if window.anchor_center is not None:
            u, v = (min(max(c, 0.0), 1.0 - 1e-9) for c in window.anchor_center)
            self.v[
                _ANCHOR
                + int(v * ANCHOR_BINS) * ANCHOR_BINS
                + int(u * ANCHOR_BINS)
            ] += 1

    def anchor_histogram(self) -> np.ndarray:
        """8x8 counts of anchor centres in tile coordinates."""
        return self.v[_ANCHOR:_OBJECTS].reshape(ANCHOR_BINS, ANCHOR_BINS)

    def objects_histogram(self) -> np.ndarray:
        """Counts of boxes per tile over ``OBJECT_EDGES``."""
        return self.v[_OBJECTS:_SCALE].copy()

    def scale_histogram(self) -> np.ndarray:
        """Counts of ``ln(1/s)``, the apparent object scale, over
        ``SCALE_EDGES`` for resampled windows; native windows count as
        ``native`` instead.
        """
        return self.v[_SCALE : self.SIZE].copy()

    def summary(self) -> dict[str, float]:
        """Per-epoch shares and rates (§3.10)."""
        g = lambda key: float(self.v[self.index(key)])  # noqa: E731
        ratio = lambda a, b: a / b if b else 0.0  # noqa: E731
        windows, non_full = g("windows"), g("windows") - g("full_frame")
        return {
            "positive_share": ratio(g("positive"), windows),
            "background_share": ratio(g("background"), windows),
            "full_frame_share": ratio(g("full_frame"), windows),
            "empty_rate": ratio(g("empty"), windows),
            "fallback_rate": ratio(g("background_fallback"), windows),
            "clipped_share": ratio(g("clipped"), g("boxes")),
            "native_share": ratio(g("native"), non_full),
            "boxes_per_tile": ratio(g("boxes"), windows),
            "objects_per_positive_tile": ratio(
                g("positive_boxes"), g("positive")
            ),
            "decodes_per_sample": ratio(g("decodes"), g("samples")),
            "dropped_visibility_per_tile": ratio(
                g("dropped_visibility"), windows
            ),
            "dropped_min_px_per_tile": ratio(g("dropped_min_px"), windows),
            "dropped_cap_per_tile": ratio(g("dropped_cap"), windows),
        }
