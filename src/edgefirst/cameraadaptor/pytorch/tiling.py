# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""PyTorch helpers for tiled full-frame evaluation (TILING.md §5, §6.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from edgefirst.cameraadaptor.tiling import (
    TargetRules,
    TilePlacement,
    crop_targets,
    lift_tile_boxes,
    merge_tiled_detections,
    tile_grid,
)


@dataclass(frozen=True)
class FramePlan:
    """A frame's deployment grid and its offset in the validator input."""

    frame_hw: tuple[int, int]
    pad: tuple[int, int]
    tiles: list[TilePlacement]


def plan_frame(
    frame_hw: Sequence[int],
    pad_xy: Sequence[int],
    tile_hw: int | Sequence[int],
    min_overlap: float,
) -> FramePlan:
    """Grid for a native frame placed at ``pad_xy`` (left, top) inside the
    validator input.

    ``tile_hw`` is the tile ``(H, W)``, or an int for a square tile.
    """
    frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
    tile_hw = (
        (tile_hw, tile_hw)
        if isinstance(tile_hw, int)
        else (int(tile_hw[0]), int(tile_hw[1]))
    )
    return FramePlan(
        frame_hw,
        (int(pad_xy[0]), int(pad_xy[1])),
        tile_grid(frame_hw, tile_hw, min_overlap),
    )


def render_tiles(
    img: torch.Tensor, plan: FramePlan, pad_value: float = 114 / 255
) -> torch.Tensor:
    """Cut every grid tile from a ``[1, C, Hp, Wp]`` validator input.

    A grid crop is never larger than its tile, so a crop that does not match
    the tile is only ever upscaled: into its letterbox content, or onto the
    whole tile when ``content`` is ``None``.
    """
    (h, w), (px, py) = plan.frame_hw, plan.pad
    frame = img[0, :, py : py + h, px : px + w]
    tiles = []
    for p in plan.tiles:
        x, y = p.origin
        cw, ch = p.crop_size
        th, tw = p.tile_hw
        crop = frame[:, y : y + ch, x : x + cw]
        if p.content is None and (cw, ch) == (tw, th):
            tiles.append(crop)
            continue
        left, top, nw, nh = p.content or (0, 0, tw, th)
        tile = crop.new_full((crop.shape[0], th, tw), pad_value)
        tile[:, top : top + nh, left : left + nw] = F.interpolate(
            crop[None], size=(nh, nw), mode="bilinear", align_corners=False
        )[0]
        tiles.append(tile)
    return torch.stack(tiles)


def tile_targets(
    frame_boxes: np.ndarray,
    classes: np.ndarray,
    plan: FramePlan,
    rules: TargetRules,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tile targets in Ultralytics batch layout (``batch_idx``, ``cls``
    [n,1], normalised xywh).

    ``truncation="ignore"`` is rejected: its ignore regions have no place in
    this layout and would be scored as background.
    """
    if rules.truncation == "ignore":
        raise ValueError(
            "tile_targets does not support targets.truncation 'ignore'"
        )
    idx, cls, boxes = [], [], []
    for i, p in enumerate(plan.tiles):
        r = crop_targets(
            p.rect,
            p.tile_hw,
            {"boxes": frame_boxes, "classes": classes},
            rules,
            p.content,
        )
        th, tw = p.tile_hw
        b = r.boxes
        idx += [float(i)] * len(b)
        cls += r.classes.tolist()
        boxes.append(
            np.stack(
                [
                    (b[:, 0] + b[:, 2]) / 2 / tw,
                    (b[:, 1] + b[:, 3]) / 2 / th,
                    (b[:, 2] - b[:, 0]) / tw,
                    (b[:, 3] - b[:, 1]) / th,
                ],
                axis=1,
            ).reshape(-1, 4)
        )
    return (
        torch.tensor(idx, dtype=torch.float32),
        torch.tensor(cls, dtype=torch.float32).reshape(-1, 1),
        torch.from_numpy(np.concatenate(boxes).astype(np.float32)),
    )


def merge_frame(
    per_tile: list[dict],
    plan: FramePlan,
    max_det: int = 300,
    device: str | torch.device = "cpu",
) -> dict:
    """Lift per-tile detections, merge across tiles and return validator-input
    pixels.
    """
    boxes, scores, classes = [], [], []
    for p, pred in zip(plan.tiles, per_tile, strict=True):
        th, tw = p.tile_hw
        norm = pred["bboxes"].float().cpu().numpy() / np.array(
            [tw, th, tw, th], np.float32
        )
        boxes.append(lift_tile_boxes(norm, p))
        scores.append(pred["conf"].float().cpu().numpy())
        classes.append(pred["cls"].cpu().numpy())
    b, s, c = merge_tiled_detections(
        np.concatenate(boxes),
        np.concatenate(scores),
        np.concatenate(classes).astype(np.int64),
        max_det=max_det,
    )
    (h, w), (px, py) = plan.frame_hw, plan.pad
    b = np.clip(b, 0, np.array([w, h, w, h], np.float32)) + np.array(
        [px, py, px, py], np.float32
    )
    return {
        "bboxes": torch.from_numpy(b).to(device),
        "conf": torch.from_numpy(s).to(device),
        "cls": torch.from_numpy(c.astype(np.float32)).to(device),
        "extra": torch.zeros((len(b), 0), device=device),
    }
