# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Multi-label NMS of the §5 validation protocol, as a NumPy reference.

It reproduces Ultralytics ``non_max_suppression(multi_label=True)`` as its
validator calls it, so a framework without Ultralytics (ModelPack) can score
under the same protocol: one candidate per (box, class) pair scoring above the
floor, the top ``max_candidates`` by score, class-aware greedy suppression of
any box whose IoU with a kept box exceeds ``iou``, and at most ``max_det``
detections.
"""

from __future__ import annotations

import numpy as np

from ._types import ArrayLike

PROTOCOL_IOU_WHOLE_FRAME = 0.7
PROTOCOL_IOU_PER_TILE = 0.5
PROTOCOL_SCORE_FLOOR = 0.001
PROTOCOL_MAX_CANDIDATES = 30000


def _iou_one(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    iw = np.minimum(box[2], others[:, 2]) - np.maximum(box[0], others[:, 0])
    ih = np.minimum(box[3], others[:, 3]) - np.maximum(box[1], others[:, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area + areas - inter)


def _greedy(boxes: np.ndarray, iou: float, limit: int) -> list[int]:
    """Indices kept from ``boxes`` already in descending score order, at
    most ``limit`` of them.
    """
    alive = np.ones(len(boxes), dtype=bool)
    keep = []
    for i in range(len(boxes)):
        if len(keep) == limit:
            break
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.flatnonzero(alive[i + 1 :]) + i + 1
        if len(rest):
            alive[rest[_iou_one(boxes[i], boxes[rest]) > iou]] = False
    return keep


def nms_multilabel(
    boxes_xyxy: ArrayLike,
    scores: ArrayLike,
    iou: float = PROTOCOL_IOU_WHOLE_FRAME,
    score_threshold: float = PROTOCOL_SCORE_FLOOR,
    max_det: int = 300,
    max_candidates: int = PROTOCOL_MAX_CANDIDATES,
    class_agnostic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detections of one image from ``boxes_xyxy`` ``[A, 4]`` and per-class
    ``scores`` ``[A, C]``.

    Returns ``(boxes [n, 4] float32, scores [n] float32, classes [n] int64)``
    in descending score order, ``n <= max_det``. Equal scores keep candidate
    order (box, then class).
    """
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    if not len(b):
        return b, np.zeros(0, np.float32), np.zeros(0, np.int64)
    s = np.asarray(scores, dtype=np.float32).reshape(len(b), -1)
    anchor, cls = np.nonzero(s > np.float32(score_threshold))
    conf = s[anchor, cls]
    order = np.argsort(-conf, kind="stable")
    if len(order) > max_candidates:
        order = order[:max_candidates]
    anchor, cls, conf = anchor[order], cls[order], conf[order]
    kept = []
    groups = [np.arange(len(conf))] if class_agnostic else None
    if groups is None:
        groups = [np.flatnonzero(cls == k) for k in np.unique(cls)]
    # A group keeps at most max_det: no more of it can reach the final top
    # max_det, and suppression only looks at boxes kept earlier.
    for members in groups:
        kept += members[_greedy(b[anchor[members]], iou, max_det)].tolist()
    kept = np.asarray(sorted(kept), dtype=np.int64)[:max_det]
    return b[anchor[kept]], conf[kept], cls[kept].astype(np.int64)


__all__ = [
    "PROTOCOL_IOU_PER_TILE",
    "PROTOCOL_IOU_WHOLE_FRAME",
    "PROTOCOL_MAX_CANDIDATES",
    "PROTOCOL_SCORE_FLOOR",
    "nms_multilabel",
]
