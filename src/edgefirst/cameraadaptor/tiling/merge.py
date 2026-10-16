# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Lift and cross-tile merge reference (TILING.md §6.4), matching HAL
``edgefirst-decoder``.
"""

from __future__ import annotations

import numpy as np

from ._types import ArrayLike
from .grid import TilePlacement

# Classes with at most this many candidates precompute all pairwise overlaps;
# larger ones compute
# a row only for each kept box, which is at most max_det rows.
_DENSE_MAX = 256


def lift_tile_boxes(
    boxes_norm: ArrayLike, placement: TilePlacement
) -> np.ndarray:
    """Lift model-input-normalised xyxy boxes into frame pixels.

    Float32 throughout. A letterboxed placement is un-padded and clamped to its
    content first. Results are not clipped to the frame, as in HAL.
    """
    b = np.asarray(boxes_norm, dtype=np.float32).reshape(-1, 4)
    x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if placement.letterbox is not None:
        x0, x1 = np.minimum(x0, x1), np.maximum(x0, x1)
        y0, y1 = np.minimum(y0, y1), np.maximum(y0, y1)
        lx0, ly0, lx1, ly1 = (np.float32(v) for v in placement.letterbox)
        one = np.float32(1.0)
        inv_w = one / (lx1 - lx0) if lx1 > lx0 else one
        inv_h = one / (ly1 - ly0) if ly1 > ly0 else one
        x0, x1 = (np.clip((v - lx0) * inv_w, 0, 1) for v in (x0, x1))
        y0, y1 = (np.clip((v - ly0) * inv_h, 0, 1) for v in (y0, y1))
    ox, oy = (np.float32(v) for v in placement.origin)
    cw, ch = (np.float32(v) for v in placement.crop_size)
    return np.stack(
        [ox + x0 * cw, oy + y0 * ch, ox + x1 * cw, oy + y1 * ch], axis=1
    ).astype(np.float32)


def _ordered_corners(b: np.ndarray) -> np.ndarray:
    """xyxy with x0 <= x1 and y0 <= y1."""
    return np.stack(
        [
            np.minimum(b[:, 0], b[:, 2]),
            np.minimum(b[:, 1], b[:, 3]),
            np.maximum(b[:, 0], b[:, 2]),
            np.maximum(b[:, 1], b[:, 3]),
        ],
        axis=1,
    )


def _class_groups(c: np.ndarray, class_agnostic: bool):
    """Merge groups, and each box's group and index within it."""
    n = len(c)
    groups = (
        [np.arange(n)]
        if class_agnostic
        else [np.flatnonzero(c == k) for k in np.unique(c)]
    )
    group_of = np.empty(n, dtype=np.int64)
    local = np.empty(n, dtype=np.int64)
    for g, members in enumerate(groups):
        group_of[members] = g
        local[members] = np.arange(len(members))
    return groups, group_of, local


def _union(box, score, hit_boxes, hit_scores):
    """Enclosing box of a group at the group's maximum score."""
    box = np.array(
        [
            min(box[0], hit_boxes[:, 0].min()),
            min(box[1], hit_boxes[:, 1].min()),
            max(box[2], hit_boxes[:, 2].max()),
            max(box[3], hit_boxes[:, 3].max()),
        ],
        dtype=np.float32,
    )
    return box, max(score, hit_scores.max())


def _overlap_row(dense, boxes, area, j: int, metric, thr, eps) -> np.ndarray:
    """Row ``j`` of a group's overlap matrix, from ``dense`` when the group
    is small enough to precompute it.
    """
    if dense is not None:
        return dense[j]
    return _overlaps(boxes, area, metric, thr, eps, (j, j + 1))[0]


def merge_tiled_detections(
    boxes: ArrayLike,
    scores: ArrayLike,
    classes: ArrayLike,
    *,
    metric: str = "ios",
    threshold: float = 0.5,
    mode: str = "keep_best",
    class_agnostic: bool = False,
    max_det: int = 300,
    score_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy merge of lifted detections in frame pixels.

    Descending score order, ties by input order. Each kept box claims every
    later unclaimed box of its class (unless ``class_agnostic``) whose
    ``metric`` against the kept box is at least ``threshold``. ``keep_best``
    emits the kept box unchanged; ``union`` emits the group's enclosing box at
    the group's maximum score. IoS is intersection over the smaller area.

    A box only claims boxes after it in score order, so the loop stops once
    ``max_det`` boxes are kept or scores fall below ``score_threshold``; the
    output is the same as merging every box and truncating.
    """
    if metric not in ("ios", "iou") or mode not in ("keep_best", "union"):
        raise ValueError(f"unsupported merge metric/mode: {metric}/{mode}")
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    c = np.asarray(classes).reshape(-1)
    if len(b) == 0:
        return b, s, c
    b = _ordered_corners(b)
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    order = np.lexsort((np.arange(len(b)), -s))
    thr, eps = np.float32(threshold), np.float32(1e-9)
    groups, group_of, local = _class_groups(c, class_agnostic)
    boxes_of = [b[m] for m in groups]
    area_of = [area[m] for m in groups]
    hits = [
        _overlaps(bm, am, metric, thr, eps) if len(bm) <= _DENSE_MAX else None
        for bm, am in zip(boxes_of, area_of, strict=True)
    ]
    used = np.zeros(len(b), dtype=bool)
    floor = np.float32(score_threshold)
    out_b, out_s, out_c = [], [], []
    for i in order:
        if len(out_b) == max_det or (score_threshold > 0.0 and s[i] < floor):
            break
        if used[i]:
            continue
        used[i] = True
        g, j = group_of[i], local[i]
        members = groups[g]
        row = _overlap_row(
            hits[g], boxes_of[g], area_of[g], j, metric, thr, eps
        )
        # Every unclaimed box comes later in score order, so claiming all
        # unclaimed overlaps claims exactly the later ones.
        hit = members[row & ~used[members]]
        used[hit] = True
        box, score = b[i].copy(), s[i]
        if mode == "union" and len(hit):
            box, score = _union(box, score, b[hit], s[hit])
        out_b.append(box)
        out_s.append(score)
        out_c.append(c[i])
    if not out_b:
        return b[:0], s[:0], c[:0]
    return _collect(out_b, out_s, out_c, score_threshold, max_det)


def _collect(out_b, out_s, out_c, score_threshold: float, max_det: int):
    """Kept boxes as arrays, above the score floor and capped at max_det."""
    ob, os_, oc = (
        np.stack(out_b),
        np.asarray(out_s, np.float32),
        np.asarray(out_c),
    )
    if score_threshold > 0.0:
        keep = os_ >= np.float32(score_threshold)
        ob, os_, oc = ob[keep], os_[keep], oc[keep]
    return ob[:max_det], os_[:max_det], oc[:max_det]


def _overlaps(b, area, metric, thr, eps, rows=None, block=1024):
    """Boolean ``metric(i, j) >= thr`` for rows ``i`` in ``rows`` (default all)
    against every box.

    Computed in row blocks to bound memory; any row is bit-identical however it
    is computed.
    """
    lo, hi = rows if rows is not None else (0, len(b))
    out = np.empty((hi - lo, len(b)), dtype=bool)
    for r0 in range(lo, hi, block):
        r = slice(r0, min(r0 + block, hi))
        iw = np.maximum(
            np.minimum(b[r, 2:3], b[:, 2]) - np.maximum(b[r, 0:1], b[:, 0]), 0
        )
        ih = np.maximum(
            np.minimum(b[r, 3:4], b[:, 3]) - np.maximum(b[r, 1:2], b[:, 1]), 0
        )
        inter = iw * ih
        denom = (
            np.minimum(area[r, None], area)
            if metric == "ios"
            else area[r, None] + area - inter
        )
        out[r0 - lo : r.stop - lo] = inter / np.maximum(denom, eps) >= thr
    return out
