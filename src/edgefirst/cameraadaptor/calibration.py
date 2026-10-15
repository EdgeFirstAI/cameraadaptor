# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Model-free INT8 calibration sample selection (TILING.md §6.5).

Selection is a pure function of the pool's decoded source pixels, the sample
count and a seed, so every trainer that calibrates the same dataset picks the
same frames, whatever the model's input geometry or camera adaptor.

``max_dynamic_range_v1`` selects frames:

1. Describe each decoded frame: per-channel minimum and maximum, their first
   row-major positions, and a normalised 64-bin histogram
   (``image_descriptor``, assembled by ``pool_descriptors``).
2. Seed the selection with the frames holding the pool's per-channel extrema.
3. Greedily add the frame whose inclusion most reduces the L1 distance between
   the selected mean histogram and the pool mean histogram.
4. Once that gain saturates, fill the rest by a seeded random draw.
5. Return pool indices in ascending order, which is the snapshot's sample
   order.

``max_dynamic_range_tiled_v1`` calibrates a tiled model on deployment-grid
tiles of the selected frames at native scale (``plan_tile_samples``).

Decoding stays with the caller: descriptors take a decoded ``H x W x C`` uint8
array, and tiled planning takes each frame's decoded ``(H, W)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .tiling._types import HW
from .tiling.grid import TilePlacement, tile_grid

SELECTION_ALGORITHM = "max_dynamic_range_v1"
TILED_SELECTION_ALGORITHM = "max_dynamic_range_tiled_v1"
DEFAULT_SEED = 42

# Constants of max_dynamic_range_v1. Changing any of them changes the selected
# set, which needs a new algorithm name.
HIST_BINS = 64
_BIN_SHIFT = 2  # 256 uint8 levels >> 2 == 64 bins, exact integer binning
_SATURATION_EPS = 1e-12


@dataclass(frozen=True)
class ImageDescriptor:
    """Source-pixel statistics of one decoded frame, per channel."""

    mins: np.ndarray  # (C,) uint8
    maxs: np.ndarray  # (C,) uint8
    hist: np.ndarray  # (C, HIST_BINS) float64, each row sums to 1
    min_pos: np.ndarray  # (C, 2) int64, (y, x) of the first minimum
    max_pos: np.ndarray  # (C, 2) int64, (y, x) of the first maximum


def image_descriptor(image: np.ndarray) -> ImageDescriptor:
    """Describe one decoded ``H x W x C`` uint8 frame at native resolution.

    Histograms are normalised so frame area does not weight the pool. Pass
    frames in the colour order the trainer decodes them (OpenCV BGR for both
    EdgeFirst trainers).
    """
    img = np.asarray(image)
    if img.dtype != np.uint8 or img.ndim != 3:
        raise ValueError(
            f"expected an H x W x C uint8 frame, got {img.dtype} {img.shape}"
        )
    h, w, channels = img.shape
    flat = img.reshape(-1, channels)
    hist = np.empty((channels, HIST_BINS), dtype=np.float64)
    for c in range(channels):
        counts = np.bincount(flat[:, c] >> _BIN_SHIFT, minlength=HIST_BINS)
        hist[c] = counts / (h * w)
    first = lambda idx: np.array(  # noqa: E731
        [divmod(int(i), w) for i in idx], dtype=np.int64
    )
    return ImageDescriptor(
        flat.min(axis=0).astype(np.uint8),
        flat.max(axis=0).astype(np.uint8),
        hist,
        first(flat.argmin(axis=0)),
        first(flat.argmax(axis=0)),
    )


def pool_descriptors(
    descriptors: Sequence[ImageDescriptor | None],
) -> dict:
    """Stack per-frame descriptors into the pool arrays selection reads.

    ``None`` marks a frame that could not be decoded; it keeps its index, so
    selected indices still name pool frames, but is never selected. So is a
    frame whose channel count differs from the first decoded frame's.

    Returns a dict of ``mins``/``maxs`` (N, C) uint8, ``hists`` (N, C, 64),
    ``valid`` (N,) bool, ``min_pos``/``max_pos`` (N, C, 2), ``num_images``
    and ``channels``.
    """
    n = len(descriptors)
    if n == 0:
        raise ValueError("calibration pool is empty")
    channels = next((d.hist.shape[0] for d in descriptors if d is not None), 3)
    pool = {
        "mins": np.full((n, channels), 255, dtype=np.uint8),
        "maxs": np.zeros((n, channels), dtype=np.uint8),
        "hists": np.zeros((n, channels, HIST_BINS), dtype=np.float64),
        "min_pos": np.zeros((n, channels, 2), dtype=np.int64),
        "max_pos": np.zeros((n, channels, 2), dtype=np.int64),
        "valid": np.zeros(n, dtype=bool),
        "num_images": n,
        "channels": channels,
    }
    for i, d in enumerate(descriptors):
        if d is None or d.hist.shape[0] != channels:
            continue
        pool["mins"][i], pool["maxs"][i] = d.mins, d.maxs
        pool["hists"][i] = d.hist
        pool["min_pos"][i], pool["max_pos"][i] = d.min_pos, d.max_pos
        pool["valid"][i] = True
    if not pool["valid"].any():
        raise ValueError("no decodable frames in the calibration pool")
    return pool


def _extremum_seeds(pool: Mapping) -> list[tuple[int, str, int]]:
    """Per channel, the first valid frame holding the pool's minimum and
    maximum: ``[(index, "min" | "max", channel)]``.
    """
    mins, maxs, valid = pool["mins"], pool["maxs"], pool["valid"]
    mins_masked = np.where(
        valid[:, None], mins.astype(np.int16), np.int16(256)
    )
    maxs_masked = np.where(valid[:, None], maxs.astype(np.int16), np.int16(-1))
    seeds = []
    for c in range(mins.shape[1]):
        seeds.append((int(np.argmin(mins_masked[:, c])), "min", c))
        seeds.append((int(np.argmax(maxs_masked[:, c])), "max", c))
    return seeds


def extremum_positions(
    pool: Mapping, selected: Sequence[int]
) -> dict[int, list[tuple[int, int]]]:
    """``(y, x)`` of the pool's per-channel extrema in the selected frames
    that were seeded for them.
    """
    chosen = {int(i) for i in selected}
    positions: dict[int, list[tuple[int, int]]] = {}
    for index, kind, c in _extremum_seeds(pool):
        if index in chosen:
            y, x = pool[f"{kind}_pos"][index, c]
            positions.setdefault(index, []).append((int(y), int(x)))
    return positions


def select_max_dynamic_range(
    pool: Mapping, count: int, seed: int = DEFAULT_SEED
) -> list[int]:
    """Pool indices maximising dynamic-range coverage, ascending.

    ``count`` is clamped to the number of valid frames.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    hists, valid = pool["hists"], pool["valid"]
    if count >= int(valid.sum()):
        return [int(i) for i in np.flatnonzero(valid)]
    target = hists[valid].mean(axis=0)
    selected = np.zeros(len(valid), dtype=bool)
    for index, _, _ in _extremum_seeds(pool):
        selected[index] = True
    seeded = np.flatnonzero(selected)
    if len(seeded) > count:
        selected[:] = False
        selected[seeded[:count]] = True

    hist_sum = hists[selected].sum(axis=0)
    n_sel = int(selected.sum())
    while n_sel < count:
        remaining = valid & ~selected
        if not remaining.any():
            break
        current = (
            np.abs(hist_sum / n_sel - target).sum()
            if n_sel
            else np.abs(target).sum()
        )
        cand = np.flatnonzero(remaining)
        means = (hist_sum[None] + hists[cand]) / (n_sel + 1)
        gains = current - np.abs(means - target).sum(axis=(1, 2))
        best = int(np.argmax(gains))
        if gains[best] <= _SATURATION_EPS:
            break
        pick = int(cand[best])
        selected[pick] = True
        hist_sum += hists[pick]
        n_sel += 1

    if n_sel < count:
        remaining = sorted(int(i) for i in np.flatnonzero(valid & ~selected))
        rng = np.random.default_rng(seed)
        selected[rng.choice(remaining, size=count - n_sel, replace=False)] = (
            True
        )
    return [int(i) for i in np.flatnonzero(selected)]


@dataclass(frozen=True)
class TileSample:
    """One tiled-calibration sample: a grid tile of pool frame ``index``, or
    the whole frame when ``placement`` is ``None``.
    """

    index: int
    placement: TilePlacement | None


def _tile_at(grid: Sequence[TilePlacement], y: int, x: int) -> int:
    return next(
        p.index
        for p in grid
        if p.origin[0] <= x < p.origin[0] + p.crop_size[0]
        and p.origin[1] <= y < p.origin[1] + p.crop_size[1]
    )


def plan_tile_samples(
    frame_hw: Mapping[int, HW] | Sequence[HW],
    selected: Sequence[int],
    tile_hw: HW,
    min_overlap: float,
    extrema: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    full_frame: bool = False,
    seed: int = DEFAULT_SEED,
    count: int | None = None,
) -> list[TileSample]:
    """Samples of a tiled snapshot, in pool order (§6.5).

    ``frame_hw[index]`` is pool frame ``index``'s decoded ``(H, W)``;
    ``selected`` is taken in ascending pool order whatever its order. A frame
    seeded for pool extrema contributes the lowest-index grid tile holding each
    extremum pixel, once per tile; any other frame contributes one tile drawn
    by a generator seeded from ``(seed, index)``. When those number fewer than
    ``count``, frames add further distinct tiles, in an order drawn from the
    same generator, one per frame per round (fewest first), until ``count``
    tiles are planned or every tile is taken. With ``full_frame`` each frame
    then adds the whole frame.
    """
    tile = (int(tile_hw[0]), int(tile_hw[1]))
    frames = []
    for index in sorted({int(i) for i in selected}):
        h, w = frame_hw[index]
        grid = tile_grid((int(h), int(w)), tile, min_overlap)
        rng = np.random.default_rng([seed, index])
        points = (extrema or {}).get(index)
        if points:
            first = sorted({_tile_at(grid, y, x) for y, x in points})
        else:
            first = [int(rng.integers(len(grid)))]
        rest = [
            t for t in rng.permutation(len(grid)).tolist() if t not in first
        ]
        frames.append((index, grid, first, rest))

    taken = [len(first) for _, _, first, _ in frames]
    extra = (count or 0) - sum(taken)
    while extra > 0:
        open_ = [i for i, f in enumerate(frames) if taken[i] < len(f[1])]
        if not open_:
            break
        for i in sorted(open_, key=lambda i: taken[i])[:extra]:
            taken[i] += 1
            extra -= 1

    samples = []
    for (index, grid, first, rest), n in zip(frames, taken, strict=True):
        chosen = sorted(first + rest[: n - len(first)])
        samples.extend(TileSample(index, grid[t]) for t in chosen)
        if full_frame:
            samples.append(TileSample(index, None))
    return samples


__all__ = [
    "DEFAULT_SEED",
    "HIST_BINS",
    "SELECTION_ALGORITHM",
    "TILED_SELECTION_ALGORITHM",
    "ImageDescriptor",
    "TileSample",
    "extremum_positions",
    "image_descriptor",
    "plan_tile_samples",
    "pool_descriptors",
    "select_max_dynamic_range",
]
