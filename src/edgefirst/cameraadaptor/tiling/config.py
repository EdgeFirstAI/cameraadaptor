# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Tiling configuration (TILING.md §3.9). Parameter names are normative."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SPEC_VERSION = "3.3"
KINDS = ("positive", "background", "full_frame")


def _pair(value) -> tuple[int, int]:
    if isinstance(value, (int, float)):
        return int(value), int(value)
    h, w = value
    return int(h), int(w)


@dataclass(frozen=True)
class Mix:
    """Per-window kind probabilities (§3.2), normalised per frame."""

    positive: float = 0.75
    background: float = 0.15
    full_frame: float = 0.10

    def shares(
        self, has_objects: bool = True, final: bool = False
    ) -> tuple[float, float, float]:
        """Normalised (positive, background, full_frame) for one frame.

        The final phase moves the full-frame share to positive (§3.6). A frame
        without ground truth moves the positive share to background (§3.2).
        """
        pos, bg, ff = self.positive, self.background, self.full_frame
        if final:
            pos, ff = pos + ff, 0.0
        if not has_objects:
            pos, bg = 0.0, bg + pos
        total = pos + bg + ff
        if total <= 0.0:
            raise ValueError("tiling mix shares sum to zero for this frame")
        return pos / total, bg / total, ff / total


@dataclass(frozen=True)
class ScaleConfig:
    """Window scale rule (§3.6)."""

    native_prob: float = 0.5
    range: tuple[float, float] = (0.75, 1.333)


@dataclass(frozen=True)
class TargetRules:
    """Target rules applied by ``crop_targets`` (§3.8)."""

    min_visibility: float = 0.25
    min_box_px: float = 2.0
    truncation: str = "drop"
    ignore_visibility: float = 0.05
    max_boxes: int | None = None


_NESTED = {"mix": Mix, "scale": ScaleConfig, "targets": TargetRules}


def _nest(data: Mapping[str, Any]) -> dict[str, Any]:
    """Fold dotted keys (``"mix.positive"``) into nested mappings."""
    nested: dict[str, Any] = {}
    for key, value in data.items():
        head, _, tail = key.partition(".")
        if tail:
            nested.setdefault(head, {})[tail] = value
        elif isinstance(value, Mapping):
            nested.setdefault(key, {}).update(value)
        else:
            nested[key] = value
    return nested


def _unknown_keys(nested: Mapping[str, Any], names: set[str]) -> list[str]:
    unknown = sorted(set(nested) - names)
    for key, sub in _NESTED.items():
        if key in nested:
            allowed = {f.name for f in fields(sub)}
            unknown += sorted(f"{key}.{k}" for k in set(nested[key]) - allowed)
    return unknown


def _field(key: str, value: Any) -> Any:
    """Constructor value for one top-level field."""
    if key == "tile_size":
        return _pair(value)
    if key not in _NESTED:
        return value
    if key == "scale" and "range" in value:
        value = {**value, "range": tuple(float(v) for v in value["range"])}
    return _NESTED[key](**value)


@dataclass(frozen=True)
class TilingConfig:
    """The §3.9 configuration shared by every trainer.

    ``grid_min_overlap`` is the EvenDist ``min_overlap`` used for grid snapping
    (§3.3, §3.4); it is the value written to the metadata ``grid`` block.
    """

    enabled: bool = False
    tile_size: tuple[int, int] = (640, 640)
    frames_per_batch: int = 4
    epoch_mode: str = "frames"
    mix: Mix = field(default_factory=Mix)
    grid_prob: float = 0.25
    grid_min_overlap: float = 0.1
    cut_prob: float = 0.1
    placement_candidates: int = 4
    anchor_weighting: str = "uniform"
    background_max_tries: int = 16
    hard_fraction: float = 0.5
    scale: ScaleConfig = field(default_factory=ScaleConfig)
    resize: str = "antialias"
    final_phase_epochs: int = 10
    targets: TargetRules = field(default_factory=TargetRules)
    class_balance: str = "none"
    rfs_t: float = 0.001
    seed: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> TilingConfig:
        """Build from nested mappings or dotted keys (``"mix.positive"``)."""
        nested = _nest(data or {})
        unknown = _unknown_keys(nested, {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown tiling parameters: {unknown}")
        cfg = cls(**{key: _field(key, v) for key, v in nested.items()})
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dictionary; ``from_dict(to_dict())`` round-trips."""
        data = asdict(self)
        data["tile_size"] = list(self.tile_size)
        data["scale"]["range"] = list(self.scale.range)
        return data

    def validate(self) -> None:
        """Raise ValueError when a parameter is out of range."""

        def check(ok: bool, message: str) -> None:
            if not ok:
                raise ValueError(f"invalid tiling config: {message}")

        check(
            self.tile_size[0] > 0 and self.tile_size[1] > 0,
            "tile_size must be positive",
        )
        check(self.frames_per_batch >= 1, "frames_per_batch must be >= 1")
        check(
            self.placement_candidates >= 1, "placement_candidates must be >= 1"
        )
        check(
            self.epoch_mode in ("frames", "tiles"),
            "epoch_mode must be frames or tiles",
        )
        shares = (self.mix.positive, self.mix.background, self.mix.full_frame)
        check(min(shares) >= 0.0, "mix shares must be >= 0")
        check(sum(shares) > 0.0, "mix shares sum to zero")
        for name in ("grid_prob", "cut_prob", "hard_fraction"):
            check(
                0.0 <= getattr(self, name) <= 1.0, f"{name} must be in [0, 1]"
            )
        check(
            0.0 <= self.grid_min_overlap < 1.0,
            "grid_min_overlap must be in [0, 1)",
        )
        check(
            self.anchor_weighting in ("uniform", "class_balanced"),
            "anchor_weighting",
        )
        check(
            self.background_max_tries >= 1, "background_max_tries must be >= 1"
        )
        check(
            0.0 <= self.scale.native_prob <= 1.0,
            "scale.native_prob must be in [0, 1]",
        )
        lo, hi = self.scale.range
        check(0.0 < lo <= 1.0 <= hi, "scale.range must bracket 1")
        check(
            self.resize in ("antialias", "bilinear"),
            "resize must be antialias or bilinear",
        )
        check(self.final_phase_epochs >= 0, "final_phase_epochs must be >= 0")
        t = self.targets
        check(
            0.0 <= t.min_visibility <= 1.0,
            "targets.min_visibility must be in [0, 1]",
        )
        check(t.min_box_px >= 0.0, "targets.min_box_px must be >= 0")
        check(t.truncation in ("drop", "keep", "ignore"), "targets.truncation")
        check(
            0.0 <= t.ignore_visibility <= t.min_visibility,
            "targets.ignore_visibility",
        )
        check(
            t.max_boxes is None or t.max_boxes >= 1,
            "targets.max_boxes must be >= 1",
        )
        check(
            self.class_balance in ("none", "rfs"),
            "class_balance must be none or rfs",
        )
        check(self.rfs_t > 0.0, "rfs_t must be > 0")
