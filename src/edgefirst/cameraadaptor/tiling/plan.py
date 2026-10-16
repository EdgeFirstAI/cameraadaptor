# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Epoch plan for frames-mode tile batches (TILING.md §4.2).

Every frame is visited once per epoch. Frames are shuffled within the
ground-truth and background strata and interleaved evenly, then grouped
``frames_per_batch`` at a time so each group carries the dataset's frame ratio;
each group fills one batch of ``batch`` tiles. Kinds are allocated per batch so
the epoch tracks ``mix`` across all tiles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ._types import HW, ArrayLike
from .config import TilingConfig
from .grid import tile_grid
from .sampler import _hits

PLAN_STREAM = 0x504C414E


@dataclass(frozen=True)
class PlannedTile:
    """One tile of an epoch: its frame, slot within the visit and kind."""

    frame: int
    slot: int
    kind: str


@dataclass(frozen=True)
class EpochPlan:
    """Every tile of one epoch in batch order, and each batch's size."""

    tiles: tuple[PlannedTile, ...]
    batches: tuple[int, ...]
    background_frames: int
    frames: int

    @cached_property
    def _kinds_by_frame(self) -> dict[int, tuple[str, ...]]:
        by_frame: dict[int, list[str]] = {}
        for t in self.tiles:
            by_frame.setdefault(t.frame, []).append(t.kind)
        return {frame: tuple(kinds) for frame, kinds in by_frame.items()}

    def kinds(self, frame: int) -> tuple[str, ...]:
        """Kinds of ``frame``'s tiles in slot order; empty if it has none."""
        return self._kinds_by_frame.get(int(frame), ())

    def summary(self) -> dict[str, float]:
        """Planned allocation statistics (§3.10)."""
        counts = np.bincount(
            [t.frame for t in self.tiles], minlength=self.frames
        )
        n = max(len(self.tiles), 1)
        share = lambda kind: sum(t.kind == kind for t in self.tiles) / n  # noqa: E731
        return {
            "tiles_per_frame_mean": float(counts.mean())
            if self.frames
            else 0.0,
            "tiles_per_frame_min": float(counts.min()) if self.frames else 0.0,
            "tiles_per_frame_max": float(counts.max()) if self.frames else 0.0,
            "background_frame_share": self.background_frames / self.frames
            if self.frames
            else 0.0,
            "planned_positive_share": share("positive"),
            "planned_background_share": share("background"),
            "planned_full_frame_share": share("full_frame"),
        }


def gt_extent(
    frame_hw: HW, boxes_xyxy: ArrayLike, tile_hw: HW, min_overlap: float
) -> int:
    """EvenDist grid tiles intersecting any ground-truth box; 0 for a frame
    without boxes.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return 0
    grid = tile_grid(
        (int(frame_hw[0]), int(frame_hw[1])),
        (int(tile_hw[0]), int(tile_hw[1])),
        min_overlap,
    )
    return max(1, sum(bool(_hits(g.rect, boxes).any()) for g in grid))


def _frames_per_batch(frames_per_batch: int, batch: int) -> int:
    return max(1, min(int(frames_per_batch), int(batch)))


def _cap(batch: int, f: int) -> int:
    return max(1, 2 * int(batch) // f)


def batch_layout(
    num_frames: int, batch: int, frames_per_batch: int
) -> list[int]:
    """Tiles in each batch of an epoch over ``num_frames`` frames."""
    f = _frames_per_batch(frames_per_batch, batch)
    cap = _cap(batch, f)
    groups = [f] * (num_frames // f) + (
        [num_frames % f] if num_frames % f else []
    )
    return [min(int(batch), g * cap) for g in groups]


def _spread(
    total: int, weights, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Integer counts in ``[1, cap]`` summing to ``total``, proportional to
    ``weights``.
    """
    w = np.asarray(weights, dtype=np.float64)
    counts = np.ones(len(w), dtype=np.int64)
    left = int(total) - len(w)
    if left < 0 or total > len(w) * cap:
        raise ValueError(
            f"cannot spread {total} slots over {len(w)} frames with cap {cap}"
        )
    while left > 0:
        open_ = counts < cap
        share = np.where(open_, w, 0.0)
        share = (
            share / share.sum() * left
            if share.sum() > 0
            else open_ / open_.sum() * left
        )
        add = np.minimum(np.floor(share).astype(np.int64), cap - counts)
        if add.sum() == 0:
            rem = (
                np.where(open_, share - np.floor(share), -1.0)
                + rng.random(len(w)) * 1e-9
            )
            add[int(np.argmax(rem))] = 1
        counts += add
        left -= int(add.sum())
    return counts


def _allocate(
    ext: np.ndarray, n: int, cap: int, shares, rng: np.random.Generator
) -> list[list[str]]:
    """Kinds for each frame of one batch group (spec §4.2)."""
    pos, bg, ff = shares
    gt = ext > 0
    k_gt, k_bg = int(gt.sum()), int((~gt).sum())
    kinds: list[list[str]] = [[] for _ in ext]
    if k_gt == 0:
        for i, c in enumerate(_spread(n, np.ones(k_bg), cap, rng)):
            kinds[i] = ["background"] * int(c)
        return kinds
    b = int(rng.binomial(n, bg))
    r = int(rng.binomial(n - b, ff / (pos + ff))) if pos + ff > 0 else 0
    bg_frame_slots = 0
    if k_bg:
        bg_frame_slots = min(
            max(b, k_bg, n - k_gt * cap), k_bg * cap, n - k_gt
        )
        for i, c in zip(
            np.flatnonzero(~gt),
            _spread(bg_frame_slots, np.ones(k_bg), cap, rng),
            strict=True,
        ):
            kinds[i] = ["background"] * int(c)
    total_gt = n - bg_frame_slots
    gt_bg = max(0, b - bg_frame_slots)
    gt_bg = min(gt_bg, total_gt)
    r = min(r, total_gt - gt_bg)
    pool = (
        ["background"] * gt_bg
        + ["full_frame"] * r
        + ["positive"] * (total_gt - gt_bg - r)
    )
    pool = [pool[i] for i in rng.permutation(len(pool))]
    counts = _spread(total_gt, ext[gt], cap, rng)
    start = 0
    for i, c in zip(np.flatnonzero(gt), counts, strict=True):
        kinds[i] = pool[start : start + int(c)]
        start += int(c)
    return kinds


def _stratified_order(ext: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Frame order with background frames spread evenly among ground-truth
    frames.
    """
    m = len(ext)
    bg = rng.permutation(np.flatnonzero(ext == 0))
    gt = rng.permutation(np.flatnonzero(ext > 0))
    i = np.arange(m)
    take_bg = ((i + 1) * len(bg)) // max(m, 1) > (i * len(bg)) // max(m, 1)
    order = np.empty(m, dtype=np.int64)
    order[take_bg] = bg
    order[~take_bg] = gt
    return order


def plan_epoch(
    extents: Sequence[int],
    batch: int,
    cfg: TilingConfig,
    epoch: int,
    final: bool = False,
) -> EpochPlan:
    """Deterministic tile plan for one epoch (TILING.md §4.2).

    ``final`` applies the final-phase mixture (§3.6).
    """
    ext = np.asarray(extents, dtype=np.int64).reshape(-1)
    m = len(ext)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.seed), int(epoch), PLAN_STREAM])
    )
    order = _stratified_order(ext, rng)
    f = _frames_per_batch(cfg.frames_per_batch, batch)
    cap = _cap(batch, f)
    shares = cfg.mix.shares(final=final)
    tiles: list[PlannedTile] = []
    sizes: list[int] = []
    for start in range(0, m, f):
        group = order[start : start + f]
        n = min(int(batch), len(group) * cap)
        for frame, kinds in zip(
            group, _allocate(ext[group], n, cap, shares, rng), strict=True
        ):
            tiles.extend(
                PlannedTile(int(frame), slot, kind)
                for slot, kind in enumerate(kinds)
            )
        sizes.append(n)
    return EpochPlan(tuple(tiles), tuple(sizes), int((ext == 0).sum()), m)
