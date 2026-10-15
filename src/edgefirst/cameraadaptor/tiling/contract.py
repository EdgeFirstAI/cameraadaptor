# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Tiling deployment contract checks (user manual: models/metadata.md "Tiling"
and models/conversion/calibration.md "Geometry check").

Reference implementation for producers and consumers of the `tiling` metadata
section and of calibration snapshots; tests/fixtures/tiling_contract.json is
its conformance set.

`validate_tiling_block` is the producer check: it rejects every document a
trainer or converter must never emit, including a `tiling.version` other than
1. Runtimes are lenient instead: they ignore unknown keys and, for an
unsupported version, warn and treat the block as absent (catch
`UnsupportedVersion`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

MODES = ("tiled", "whole_frame")
STRIDE = 32
_FIT = ("letterbox",)
_WHOLE_FRAME_RESIZE = ("letterbox", "stretch")
_PER_TILE_NMS = ("class_aware", "class_agnostic", "none")
_MERGE_MODE = ("keep_best", "union")
_MERGE_METRIC = ("ios", "iou")


class ContractError(ValueError):
    """A document or snapshot violates the tiling deployment contract."""


class UnsupportedVersion(ContractError):
    """The `tiling` section declares a version this implementation does not
    support.
    """


def _items(value: Any, what: str) -> list:
    """Elements of a list, tuple, array or other sequence; strings and mappings
    are rejected.
    """
    if value is None or isinstance(value, (str, bytes, Mapping)):
        raise ContractError(f"{what} must be a sequence; got {value!r}")
    try:
        return list(value)
    except TypeError:
        raise ContractError(
            f"{what} must be a sequence; got {value!r}"
        ) from None


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_version_one(value: Any) -> bool:
    return _is_int(value) and value == 1


def _hw(value: Any, what: str) -> tuple[int, int]:
    items = _items(value, what)
    if len(items) != 2 or not all(_is_int(v) for v in items):
        raise ContractError(f"{what} must be [H, W] integers; got {value!r}")
    h, w = (int(v) for v in items)
    if h <= 0 or w <= 0:
        raise ContractError(f"{what} must be positive; got {value!r}")
    return h, w


def _one_of(value: Any, allowed: tuple[str, ...], what: str) -> None:
    if value not in allowed:
        raise ContractError(
            f"{what} must be one of {', '.join(allowed)}; got {value!r}"
        )


def _mapping(parent: Mapping, key: str, what: str) -> Mapping:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ContractError(
            f"{what} is required and must be a mapping; got {value!r}"
        )
    return value


def _fraction(value: Any, what: str, *, below_one: bool = False) -> None:
    ok = (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and 0 <= value <= 1
    )
    if ok and below_one and value >= 1:
        ok = False
    if not ok:
        bound = "[0, 1)" if below_one else "[0, 1]"
        raise ContractError(
            f"{what} must be a number in {bound}; got {value!r}"
        )


def _flag(value: Any, what: str) -> None:
    if not isinstance(value, bool):
        raise ContractError(f"{what} must be true or false; got {value!r}")


def _text(value: Any, what: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(
            f"{what} must be a non-empty string; got {value!r}"
        )


def _session(value: Any, what: str) -> None:
    try:
        parse_session_id(value)
    except ContractError as error:
        raise ContractError(f"{what}: {error}") from None


def model_input_hw(shape: Sequence[int]) -> tuple[int, int]:
    """Input H, W from a 4-D `input.shape` in NCHW or NHWC (channels 1-4)."""
    dims = _items(shape, "input.shape")
    if len(dims) != 4 or not all(_is_int(v) for v in dims):
        raise ContractError(
            f"input.shape must be 4 integer dimensions; got {shape!r}"
        )
    _, a, b, c = (int(v) for v in dims)
    if a <= 4 < b:
        return b, c
    if c <= 4 < a:
        return a, b
    raise ContractError(
        f"cannot tell the layout of input.shape {[int(v) for v in dims]}"
    )


def _check_grid(grid: Any, what: str) -> None:
    """EvenDist version 1 with ``min_overlap`` in [0, 1)."""
    if (
        not isinstance(grid, Mapping)
        or grid.get("algorithm") != "evendist"
        or not _is_version_one(grid.get("version"))
    ):
        shown = dict(grid) if isinstance(grid, Mapping) else grid
        raise ContractError(
            f"{what} must be evendist version 1; got {shown!r}"
        )
    _fraction(grid.get("min_overlap"), f"{what}.min_overlap", below_one=True)


def _check_version(block: Mapping) -> None:
    version = block.get("version")
    if not _is_int(version):
        raise ContractError(
            f"tiling.version must be an integer; got {version!r}"
        )
    if not _is_version_one(version):
        raise UnsupportedVersion(
            f"tiling.version {version!r} is not supported (expected 1)"
        )


def _check_geometry(block: Mapping, metadata: Mapping) -> tuple[int, int]:
    """Mode, tile and model input agree; returns the model input H,W."""
    mode = block.get("mode")
    _one_of(mode, MODES, "tiling.mode")
    tile = _hw(block.get("tile"), "tiling.tile")
    input_section = metadata.get("input", {})
    if not isinstance(input_section, Mapping):
        raise ContractError(
            f"input must be a mapping; got {type(input_section).__name__}"
        )
    input_hw = model_input_hw(input_section.get("shape"))
    if mode == "tiled" and input_hw != tile:
        raise ContractError(
            f"tiling.mode tiled requires input H,W {list(input_hw)} to equal "
            f"tiling.tile {list(tile)}"
        )
    if mode == "whole_frame" and (
        input_hw[0] % STRIDE or input_hw[1] % STRIDE
    ):
        raise ContractError(
            f"whole_frame input H,W {list(input_hw)} must be a multiple of "
            f"{STRIDE}"
        )
    return input_hw


def _check_pad(pad: Any) -> None:
    pad_items = _items(pad, "tiling.pad")
    if len(pad_items) != 3 or not all(
        _is_int(v) and 0 <= v <= 255 for v in pad_items
    ):
        raise ContractError(
            f"tiling.pad must be three integers in [0, 255]; got {pad!r}"
        )


def _check_postprocess(block: Mapping) -> None:
    per_tile = _mapping(block, "per_tile", "tiling.per_tile")
    _one_of(per_tile.get("nms"), _PER_TILE_NMS, "tiling.per_tile.nms")
    _fraction(per_tile.get("iou"), "tiling.per_tile.iou")
    _fraction(per_tile.get("score"), "tiling.per_tile.score")

    merge = _mapping(block, "merge", "tiling.merge")
    _one_of(merge.get("mode"), _MERGE_MODE, "tiling.merge.mode")
    _one_of(merge.get("metric"), _MERGE_METRIC, "tiling.merge.metric")
    _fraction(merge.get("threshold"), "tiling.merge.threshold")
    _flag(merge.get("class_agnostic"), "tiling.merge.class_agnostic")
    max_det = merge.get("max_det")
    if not _is_int(max_det) or max_det <= 0:
        raise ContractError(
            f"tiling.merge.max_det must be a positive integer; got {max_det!r}"
        )


def _check_full_frame(block: Mapping) -> bool:
    """Returns ``full_frame.enabled``."""
    full_frame = _mapping(block, "full_frame", "tiling.full_frame")
    enabled = full_frame.get("enabled")
    _flag(enabled, "tiling.full_frame.enabled")
    mode = block.get("mode")
    if enabled and mode != "tiled":
        raise ContractError(
            f"tiling.full_frame.enabled requires mode tiled; got mode {mode}"
        )
    _fraction(
        full_frame.get("own_above_area"), "tiling.full_frame.own_above_area"
    )
    return enabled


def _check_training(block: Mapping) -> None:
    training = _mapping(block, "training", "tiling.training")
    _session(training.get("session"), "tiling.training.session")
    _text(training.get("sampler"), "tiling.training.sampler")
    _text(training.get("version"), "tiling.training.version")
    _mapping(training, "params", "tiling.training.params")


def _check_export(
    block: Mapping, input_hw: tuple[int, int], full_frame: bool
) -> None:
    export = _mapping(block, "export", "tiling.export")
    _session(export.get("session"), "tiling.export.session")
    if "weights_from" in export:
        _session(export["weights_from"], "tiling.export.weights_from")
    calibration = _mapping(export, "calibration", "tiling.export.calibration")
    mode = block.get("mode")
    expected = "tiled" if mode == "tiled" else "letterbox"
    if calibration.get("resize") != expected:
        raise ContractError(
            f"tiling.export.calibration.resize must be {expected!r} for mode "
            f"{mode}; got {calibration.get('resize')!r}"
        )
    shape = calibration.get("input_shape")
    if _hw(shape, "tiling.export.calibration.input_shape") != input_hw:
        raise ContractError(
            f"tiling.export.calibration input_shape {shape} does not match "
            f"the model input {list(input_hw)}"
        )
    _flag(
        calibration.get("full_frame"), "tiling.export.calibration.full_frame"
    )
    if calibration["full_frame"] != full_frame:
        raise ContractError(
            f"tiling.export.calibration.full_frame must equal "
            f"tiling.full_frame.enabled ({full_frame}); "
            f"got {calibration['full_frame']}"
        )


def validate_tiling_block(metadata: Mapping) -> None:
    """Raise ContractError unless the `tiling` section (if any) is one a
    producer may emit.

    In version 1 every key is required except `export.weights_from` and the
    contents of `training.params`. A version other than 1 raises
    UnsupportedVersion (a ContractError).
    """
    block = metadata.get("tiling")
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise ContractError(
            f"tiling must be a mapping; got {type(block).__name__}"
        )
    _check_version(block)
    input_hw = _check_geometry(block, metadata)
    _check_grid(_mapping(block, "grid", "tiling.grid"), "tiling.grid")
    _one_of(block.get("fit"), _FIT, "tiling.fit")
    _check_pad(block.get("pad"))
    _check_postprocess(block)
    full_frame = _check_full_frame(block)
    _check_training(block)
    _check_export(block, input_hw, full_frame)


def check_converter_tiling(edgefirst: Any) -> None:
    """Raise ContractError for a `tiling` section a converter must not copy
    into its output.

    Absence of the section is accepted; a present section must be a mapping
    with version 1 (an integer) and a known mode.
    """
    if not isinstance(edgefirst, Mapping) or "tiling" not in edgefirst:
        return
    tiling = edgefirst["tiling"]
    if not isinstance(tiling, Mapping):
        raise ContractError(
            f"model tiling section must be a mapping; got "
            f"{type(tiling).__name__}"
        )
    version = tiling.get("version")
    if not _is_version_one(version):
        raise ContractError(
            f"model tiling version {version!r} is not supported (expected 1); "
            f"re-export the model"
        )
    mode = tiling.get("mode")
    if mode not in MODES:
        raise ContractError(
            f"model tiling mode {mode!r} is not supported (expected tiled or "
            f"whole_frame); "
            f"re-export the model"
        )


def check_snapshot_params(
    params: Any, model_input_hw: Sequence[int]
) -> tuple[int, int]:
    """Check a snapshot's decoded `params` against the model input and return
    its [H, W].

    Run this before reading any sample tensor.
    """
    if not isinstance(params, Mapping):
        raise ContractError(
            "calibration snapshot params are missing or not a mapping"
        )
    if "input_shape" not in params:
        raise ContractError("calibration snapshot params lack input_shape")
    if "resize" not in params:
        raise ContractError("calibration snapshot params lack resize")
    snapshot_hw = _hw(
        params["input_shape"], "calibration snapshot input_shape"
    )
    model_hw = _hw(model_input_hw, "model input")
    if snapshot_hw != model_hw:
        raise ContractError(
            f"calibration snapshot input_shape {list(snapshot_hw)} does not "
            f"match the model input "
            f"{list(model_hw)}"
        )
    return snapshot_hw


def _check_snapshot_resize(params: Mapping, tiling_mode: str | None) -> None:
    """The snapshot's resize (and grid, when tiled) suits the model's mode."""
    resize = params["resize"]
    if tiling_mode == "tiled":
        if resize != "tiled":
            raise ContractError(
                f"calibration snapshot resize must be 'tiled' for a tiled "
                f"model; got {resize!r}"
            )
        grid = params.get("grid")
        if (
            not isinstance(grid, Mapping)
            or grid.get("algorithm") != "evendist"
        ):
            raise ContractError(
                "tiled calibration snapshot lacks a grid block with algorithm "
                "evendist"
            )
        _check_grid(grid, "calibration snapshot grid")
    elif tiling_mode == "whole_frame":
        if resize != "letterbox":
            raise ContractError(
                f"calibration snapshot resize must be 'letterbox' for a "
                f"whole_frame model; "
                f"got {resize!r}"
            )
    elif resize not in _WHOLE_FRAME_RESIZE and not (
        resize == "tiled" and isinstance(params.get("tiling"), Mapping)
    ):
        raise ContractError(
            f"calibration snapshot resize must be 'letterbox' or 'stretch' "
            f"for a non-tiled model; "
            f"got {resize!r}"
        )


def check_snapshot_geometry(
    params: Mapping,
    sample_hw: Sequence[int],
    model_input_hw: Sequence[int],
    tiling_mode: str | None,
    full_frame: bool = False,
) -> None:
    """Raise ContractError unless a calibration snapshot fits the model it
    calibrates.

    `params` is the snapshot's `__metadata__["params"]` (decoded), `sample_hw`
    the H,W of its stored samples, `model_input_hw` the model's input H,W after
    any layout mapping or explicit override, `tiling_mode` the model's
    `tiling.mode` (None without a tiling section), and `full_frame` the model's
    `tiling.full_frame.enabled`. Sizes may be any sequence of integers,
    including NumPy arrays and integer scalars.
    """
    snapshot_hw = check_snapshot_params(params, model_input_hw)
    sample = _hw(sample_hw, "calibration sample H,W")
    if sample != snapshot_hw:
        raise ContractError(
            f"calibration sample H,W {list(sample)} does not match the "
            f"snapshot input_shape "
            f"{list(snapshot_hw)}"
        )
    _check_snapshot_resize(params, tiling_mode)
    has_frames = params.get("full_frame", False)
    _flag(has_frames, "calibration snapshot full_frame")
    if has_frames != bool(full_frame):
        want = "must include" if full_frame else "must not include"
        raise ContractError(
            f"calibration snapshot {want} whole-frame samples (full_frame) "
            f"for this model; "
            f"got full_frame {has_frames!r}"
        )


def parse_session_id(value: str | int) -> int:
    """Training session ID from `t-<hex>`, a decimal string or an integer; IDs
    are positive.
    """
    if isinstance(value, bool):
        raise ContractError(f"invalid session id {value!r}")

    number = None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        if value.lower().startswith("t-"):
            tail = value[2:]
            if re.fullmatch(r"[0-9a-fA-F]+", tail):
                number = int(tail, 16)
            else:
                raise ContractError(
                    f"invalid session id {value!r} (expected t-<hex> or a "
                    f"decimal integer)"
                )
        elif re.fullmatch(r"\d+", value, re.ASCII):
            number = int(value, 10)
        else:
            raise ContractError(
                f"invalid session id {value!r} (expected t-<hex> or a decimal "
                f"integer)"
            )
    else:
        raise ContractError(
            f"invalid session id {value!r} (expected t-<hex> or a decimal "
            f"integer)"
        )

    if number <= 0:
        raise ContractError(f"invalid session id {value!r} (must be positive)")
    return number
