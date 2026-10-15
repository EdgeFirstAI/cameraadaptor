# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Pixel-free window sampler (TILING.md §3.1-§3.7, §3.11)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._types import HW, ArrayLike, Rect
from .config import KINDS, TilingConfig
from .grid import TilePlacement, fit_content, tile_grid
from .targets import CropResult, crop_targets, window_transform


@dataclass(frozen=True)
class Window:
    """One sampled window: its frame rectangle, tile placement and targets."""

    kind: str
    rect: tuple[int, int, int, int]
    out_hw: tuple[int, int]
    scale: float
    content: tuple[int, int, int, int] | None
    targets: CropResult
    grid_index: int | None = None
    anchor_center: tuple[float, float] | None = None
    fallback: bool = False
    fit: str = "letterbox"

    @property
    def boxes(self) -> np.ndarray:
        """Target boxes, xyxy in tile pixels."""
        return self.targets.boxes

    @property
    def classes(self) -> np.ndarray:
        """Target classes."""
        return self.targets.classes


def make_rng(
    seed: int, epoch: int, frame_index: int, call_index: int
) -> np.random.Generator:
    """Deterministic generator for one sampler call (§3.11)."""
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(epoch), int(frame_index), int(call_index)]
        )
    )


def repeat_factors(
    frame_classes: Sequence, num_classes: int, t: float = 0.001
) -> np.ndarray:
    """Repeat-factor-sampling weight per class, ``max(1, sqrt(t / f_c))``,
    where ``f_c`` is the fraction of frames containing class ``c`` (§3.9).
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for cls in frame_classes:
        counts[np.unique(np.asarray(cls, dtype=np.int64))] += 1
    freq = counts / max(len(frame_classes), 1)
    factors = np.sqrt(t / np.maximum(freq, 1e-12))
    return np.where(freq > 0, np.maximum(1.0, factors), 1.0)


def _hits(rect, boxes) -> np.ndarray:
    x, y, w, h = rect
    iw = np.minimum(boxes[:, 2], x + w) - np.maximum(boxes[:, 0], x)
    ih = np.minimum(boxes[:, 3], y + h) - np.maximum(boxes[:, 1], y)
    return (iw > 0) & (ih > 0)


def _visible_fraction(rect, box) -> float:
    x, y, w, h = rect
    iw = max(0.0, min(box[2], x + w) - max(box[0], x))
    ih = max(0.0, min(box[3], y + h) - max(box[1], y))
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    return iw * ih / area if area > 0 else 0.0


def _window_len(scale: float, tile: int, length: int) -> int:
    return min(max(1, int(math.floor(scale * tile + 0.5))), length)


def _axis_origin(
    b0: float,
    b1: float,
    length: int,
    c: int,
    e: float,
    rng: np.random.Generator,
) -> int:
    """Uniform integer origin over the feasible interval (§3.3); never clamped
    after the draw.
    """
    lo = max(0.0, min(b1 - c, b0) - e)
    hi = min(float(length - c), max(b0, b1 - c) + e)
    lo_i, hi_i = math.ceil(lo), math.floor(hi)
    if lo_i > hi_i:
        return int(min(max(math.floor(lo + 0.5), 0), length - c))
    return int(rng.integers(lo_i, hi_i + 1))


class _FrameSampler:
    def __init__(
        self,
        frame_hw,
        boxes,
        classes,
        cfg: TilingConfig,
        rng,
        final,
        hard_regions,
        class_repeat,
    ):
        self.fh, self.fw = int(frame_hw[0]), int(frame_hw[1])
        self.th, self.tw = cfg.tile_size
        self.boxes, self.classes = boxes, classes
        self.cfg, self.rng, self.final = cfg, rng, final
        self.hard = (
            None
            if hard_regions is None
            else np.asarray(hard_regions, np.float32).reshape(-1, 4)
        )
        self._grid: list[TilePlacement] | None = None
        self._deck: list[int] = []
        self._weights = None
        if cfg.anchor_weighting == "class_balanced" and len(boxes):
            if class_repeat is None:
                raise ValueError(
                    "anchor_weighting=class_balanced needs class_repeat (see "
                    "repeat_factors)"
                )
            self._weights = np.asarray(class_repeat, dtype=np.float64)[classes]

    def grid(self) -> list[TilePlacement]:
        if self._grid is None:
            self._grid = tile_grid(
                (self.fh, self.fw),
                (self.th, self.tw),
                self.cfg.grid_min_overlap,
            )
        return self._grid

    def scale(self) -> float:
        if self.final or self.rng.random() < self.cfg.scale.native_prob:
            return 1.0
        lo, hi = self.cfg.scale.range
        return float(math.exp(self.rng.uniform(math.log(lo), math.log(hi))))

    def next_anchor(self) -> int:
        if not self._deck:
            n = len(self.boxes)
            if self._weights is None:
                self._deck = [int(i) for i in self.rng.permutation(n)]
            else:
                p = self._weights / self._weights.sum()
                self._deck = [
                    int(i)
                    for i in self.rng.choice(n, size=n, replace=False, p=p)
                ]
        return self._deck.pop(0)

    def window(
        self, kind, rect, grid_index=None, anchor=None, fallback=False
    ) -> Window:
        tile_hw = (self.th, self.tw)
        content = fit_content(rect[2], rect[3], tile_hw)
        targets = crop_targets(
            rect,
            tile_hw,
            {"boxes": self.boxes, "classes": self.classes},
            self.cfg.targets,
            content,
        )
        center = None
        if anchor is not None:
            sx, sy, tx, ty = window_transform(rect, tile_hw, content)
            b = self.boxes[anchor]
            center = (
                float(((b[0] + b[2]) / 2 * sx + tx) / self.tw),
                float(((b[1] + b[3]) / 2 * sy + ty) / self.th),
            )
        return Window(
            kind,
            tuple(int(v) for v in rect),
            tile_hw,
            rect[2] / self.tw,
            content,
            targets,
            grid_index,
            center,
            fallback,
        )

    def _contained(self, rect, anchor: int) -> int:
        """Boxes other than ``anchor`` that ``rect`` fully contains."""
        x, y, w, h = rect
        b = self.boxes
        inside = (
            (b[:, 0] >= x)
            & (b[:, 1] >= y)
            & (b[:, 2] <= x + w)
            & (b[:, 3] <= y + h)
        )
        return int(np.sum(inside)) - int(inside[anchor])

    def _best(self, items, anchor: int, rect=lambda item: item):
        """Item whose window fully contains the most boxes besides ``anchor``;
        seeded tie-break (§3.3).

        Leaving the anchor out of the score keeps its placement, including a
        cut-extension cut, as random as with a single candidate.
        """
        if len(items) == 1:
            return items[0]
        counts = np.array(
            [self._contained(rect(item), anchor) for item in items]
        )
        top = np.flatnonzero(counts == counts.max())
        return items[int(top[self.rng.integers(len(top))])]

    def positive(self, fallback=False) -> Window:
        a = self.next_anchor()
        b = self.boxes[a]
        n = self.cfg.placement_candidates
        if self.rng.random() < self.cfg.grid_prob:
            min_visibility = self.cfg.targets.min_visibility
            cands = [
                g
                for g in self.grid()
                if (v := _visible_fraction(g.rect, b)) > 0
                and v >= min_visibility
            ]
            if cands:
                g = self._best(
                    [
                        cands[int(self.rng.integers(len(cands)))]
                        for _ in range(n)
                    ],
                    a,
                    rect=lambda g: g.rect,
                )
                return self.window("positive", g.rect, g.index, a, fallback)
        s = self.scale()
        cw, ch = (
            _window_len(s, self.tw, self.fw),
            _window_len(s, self.th, self.fh),
        )
        # The cut extension applies on an axis where the anchor fits in the
        # window; a longer anchor is already cut by a window inside its span,
        # and extending past it could leave the anchor out entirely.
        cut = self.rng.random() < self.cfg.cut_prob
        bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
        ex = 0.5 * bw if cut and bw <= cw else 0.0
        ey = 0.5 * bh if cut and bh <= ch else 0.0
        rects = []
        for _ in range(n):
            x = _axis_origin(
                float(b[0]), float(b[2]), self.fw, cw, ex, self.rng
            )
            y = _axis_origin(
                float(b[1]), float(b[3]), self.fh, ch, ey, self.rng
            )
            rects.append((x, y, cw, ch))
        return self.window(
            "positive", self._best(rects, a), anchor=a, fallback=fallback
        )

    def background(self) -> Window | None:
        cfg = self.cfg
        if self.rng.random() < cfg.grid_prob:
            empty = [
                g for g in self.grid() if not _hits(g.rect, self.boxes).any()
            ]
            if empty:
                g = empty[int(self.rng.integers(len(empty)))]
                return self.window("background", g.rect, g.index)
        if (
            self.hard is not None
            and len(self.hard)
            and self.rng.random() < cfg.hard_fraction
        ):
            r = self.hard[int(self.rng.integers(len(self.hard)))]
            s = self.scale()
            cw, ch = (
                _window_len(s, self.tw, self.fw),
                _window_len(s, self.th, self.fh),
            )
            rect = (
                _axis_origin(
                    float(r[0]), float(r[2]), self.fw, cw, 0.0, self.rng
                ),
                _axis_origin(
                    float(r[1]), float(r[3]), self.fh, ch, 0.0, self.rng
                ),
                cw,
                ch,
            )
            if not _hits(rect, self.boxes).any():
                return self.window("background", rect)
        for _ in range(cfg.background_max_tries):
            s = self.scale()
            cw, ch = (
                _window_len(s, self.tw, self.fw),
                _window_len(s, self.th, self.fh),
            )
            rect = (
                int(self.rng.integers(0, self.fw - cw + 1)),
                int(self.rng.integers(0, self.fh - ch + 1)),
                cw,
                ch,
            )
            if not _hits(rect, self.boxes).any():
                return self.window("background", rect)
        return None

    def draw(self, kind: str) -> Window:
        if kind == "positive":
            return self.positive()
        if kind == "background":
            window = self.background()
            return (
                window if window is not None else self.positive(fallback=True)
            )
        return self.window("full_frame", (0, 0, self.fw, self.fh))


def sample_windows(
    frame_hw: HW,
    boxes_xyxy: ArrayLike,
    classes: ArrayLike,
    cfg: TilingConfig,
    rng: np.random.Generator,
    k: int = 1,
    phase: str = "main",
    hard_regions: ArrayLike | None = None,
    valid_rect: Rect | None = None,
    class_repeat: ArrayLike | None = None,
    kinds: Sequence[str] | None = None,
) -> list[Window]:
    """Draw windows for one frame. ``valid_rect`` (x, y, w, h) is reserved
    (§3.4): it is checked but does not yet constrain background windows.

    ``kinds`` lists the window kind of each draw (the epoch plan, §4.2).
    Without it, ``k`` kinds are drawn from ``cfg.mix`` for this frame alone
    (§3.2).
    """
    if phase not in ("main", "final"):
        raise ValueError(f"phase must be main or final, got {phase!r}")
    if valid_rect is not None and np.asarray(valid_rect).shape != (4,):
        raise ValueError(
            f"valid_rect must be (x, y, w, h), got {valid_rect!r}"
        )
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(classes, dtype=np.int32).reshape(-1)
    final = phase == "final"
    if kinds is None:
        shares = cfg.mix.shares(has_objects=len(boxes) > 0, final=final)
        kinds = [
            KINDS[i]
            for i in rng.spawn(1)[0].choice(len(KINDS), size=k, p=shares)
        ]
    else:
        kinds = list(kinds)
        unknown = sorted(set(kinds) - set(KINDS))
        if unknown:
            raise ValueError(f"unknown window kinds: {unknown}")
        if len(boxes) == 0 and any(kind != "background" for kind in kinds):
            raise ValueError(
                "positive and full_frame windows need ground truth in the "
                "frame"
            )
    sampler = _FrameSampler(
        frame_hw, boxes, classes, cfg, rng, final, hard_regions, class_repeat
    )
    return [sampler.draw(kind) for kind in kinds]
