# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""TensorFlow ops for tiled training and validation (TILING.md §7.4).

The sampler stays the NumPy core: ``visit_fn`` wraps one frame visit for
``tf.numpy_function`` and returns static shapes, so window choice is identical
to the PyTorch trainer's. Pixels stay in the graph: ``decode_frame`` applies
EXIF orientation as OpenCV does, ``render_windows`` crops and resizes with the
§3.6 filter rule, and ``crop_targets`` applies the §3.8 rules to padded boxes
for a GPU mosaic. ``nms_multilabel`` is the §5 protocol in fixed shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

try:
    import tensorflow as tf
except ImportError as err:
    raise ImportError(
        "TensorFlow is required for this module. "
        "Install with: pip install edgefirst-cameraadaptor[tensorflow]"
    ) from err

from ..tiling import (
    KINDS,
    PROTOCOL_MAX_CANDIDATES,
    PROTOCOL_SCORE_FLOOR,
    EpochPlan,
    SamplerStats,
    TargetRules,
    TilingConfig,
    make_rng,
    pack_geometry,
    pack_targets,
    sample_windows,
)
from ..tiling._types import ArrayLike
from ..tiling.exif import orientation as exif_orientation

PAD_VALUE = 114
KIND_CODES = {kind: code for code, kind in enumerate(KINDS)}
"""Integer code of each window kind, as visits carry them."""

# --- training ---------------------------------------------------------------


def visit_arrays(plan: EpochPlan) -> tuple[np.ndarray, tf.RaggedTensor]:
    """The epoch's visits as ``frames int64[V]`` and ragged kind codes
    ``int32[V, None]``, ready for ``tf.data.Dataset.from_tensor_slices``.
    """
    visits = plan.visits()
    frames = np.asarray([v.frame for v in visits], dtype=np.int64)
    codes = tf.ragged.constant(
        [[KIND_CODES[k] for k in v.kinds] for v in visits], dtype=tf.int32
    )
    return frames, codes


def visit_fn(
    cfg: TilingConfig,
    frame_hw: ArrayLike,
    targets: Callable[[int], tuple[np.ndarray, np.ndarray]],
    max_boxes: int,
    hard_regions: Callable[[int], np.ndarray | None] | None = None,
    class_repeat: ArrayLike | None = None,
) -> Callable:
    """NumPy function for one frame visit, for ``visit_windows``.

    ``frame_hw[i]`` is frame ``i``'s decoded ``(H, W)``; ``targets(i)``
    returns its ``(boxes_xyxy, classes)`` in frame pixels. The function maps
    ``(epoch, frame, kind_codes, final)`` to ``geometry int32[n, 8]``
    (``pack_geometry``), ``targets float32[n, max_boxes, 5]``
    (``pack_targets``) and the visit's ``SamplerStats`` vector, which the
    caller sums instead of sharing state across tf.data workers. The windows
    are those of ``sample_windows`` with ``make_rng(cfg.seed, epoch, frame,
    0)``, as the PyTorch trainer draws them.

    ``cfg.targets.truncation`` must be ``drop`` or ``keep``: packed targets
    carry no ignore regions.
    """
    if cfg.targets.truncation == "ignore":
        raise ValueError(
            "visit_fn does not support targets.truncation 'ignore'; use "
            "'drop' or 'keep'"
        )
    sizes = np.asarray(frame_hw, dtype=np.int64).reshape(-1, 2)
    repeat = None if class_repeat is None else np.asarray(class_repeat)

    def visit(epoch, frame, codes, final):
        e, i = int(epoch), int(frame)
        boxes, classes = targets(i)
        windows = sample_windows(
            tuple(sizes[i]),
            boxes,
            classes,
            cfg,
            make_rng(cfg.seed, e, i, 0),
            phase="final" if bool(final) else "main",
            hard_regions=None if hard_regions is None else hard_regions(i),
            class_repeat=repeat,
            kinds=[KINDS[int(c)] for c in np.asarray(codes).reshape(-1)],
        )
        stats = SamplerStats()
        for w in windows:
            stats.record(w)
        stats.add("decodes")
        stats.add("samples", len(windows))
        geom = np.stack([pack_geometry(w) for w in windows])
        packed = np.stack(
            [pack_targets(w, w.out_hw, max_boxes) for w in windows]
        )
        return geom, packed, stats.v

    return visit


def visit_windows(
    fn: Callable,
    epoch: tf.Tensor,
    frame: tf.Tensor,
    codes: tf.Tensor,
    final: tf.Tensor,
    max_boxes: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """``tf.numpy_function`` wrapper of a ``visit_fn``, with static inner
    shapes: ``int32[n, 8]``, ``float32[n, max_boxes, 5]``,
    ``float64[SamplerStats.SIZE]``.
    """
    geom, packed, stats = tf.numpy_function(
        fn,
        [epoch, frame, codes, final],
        [tf.int32, tf.float32, tf.float64],
        stateful=False,
    )
    geom.set_shape([None, 8])
    packed.set_shape([None, max_boxes, 5])
    stats.set_shape([SamplerStats.SIZE])
    return geom, packed, stats


def _apply_orientation(image: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
    branches = [
        lambda: image,
        lambda: image,
        lambda: tf.image.flip_left_right(image),
        lambda: tf.image.rot90(image, 2),
        lambda: tf.image.flip_up_down(image),
        lambda: tf.image.transpose(image),
        lambda: tf.image.rot90(image, 3),
        lambda: tf.image.rot90(tf.image.flip_up_down(image), 1),
        lambda: tf.image.rot90(image, 1),
    ]
    return tf.switch_case(tf.clip_by_value(value, 0, 8), branches)


def _as_bytes(raw) -> bytes:
    """A string tensor's value inside ``tf.numpy_function``: it arrives as
    ``bytes`` or as a 0-d object array, depending on the TensorFlow build.
    """
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    return bytes(raw)


def decode_frame(data: tf.Tensor) -> tf.Tensor:
    """Decode an encoded frame to ``uint8[H, W, 3]`` RGB with its EXIF
    orientation applied, as OpenCV ``imread`` does (TILING.md §4.3).
    """
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    value = tf.numpy_function(
        lambda raw: np.int32(exif_orientation(_as_bytes(raw))),
        [data],
        tf.int32,
        stateful=False,
    )
    value.set_shape([])
    return _apply_orientation(image, value)


def _fit(
    crop: tf.Tensor, ch: tf.Tensor, cw: tf.Tensor, resize: str
) -> tf.Tensor:
    # Resizes read the uint8 crop directly: TensorFlow's uint8 kernels give
    # the same result as a float32 input several times faster.
    h, w = tf.shape(crop)[0], tf.shape(crop)[1]
    down = tf.logical_or(ch < h, cw < w)
    size = tf.stack([ch, cw])
    area = resize == "antialias"

    def scaled():
        out = tf.cond(
            tf.logical_and(down, area),
            lambda: tf.image.resize(crop, size, method="area"),
            lambda: tf.image.resize(crop, size, method="bilinear"),
        )
        return tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8)

    return tf.cond(
        tf.logical_and(tf.equal(ch, h), tf.equal(cw, w)),
        lambda: crop,
        scaled,
    )


def render_windows(
    image: tf.Tensor,
    geometry: tf.Tensor,
    tile_hw: Sequence[int],
    resize: str = "antialias",
    pad: int = PAD_VALUE,
) -> tf.Tensor:
    """Tiles ``uint8[n, T_h, T_w, 3]`` for packed window geometry
    ``int32[n, 8]`` (``pack_geometry`` or ``pack_placements``).

    Each crop is placed at its content rectangle on a ``pad``-filled tile. A
    crop the size of its content is copied unchanged; a downscale uses area
    filtering when ``resize`` is ``"antialias"``, and every other resize is
    bilinear (§3.6).
    """
    if resize not in ("antialias", "bilinear"):
        raise ValueError(f"resize must be antialias or bilinear, got {resize}")
    th, tw = int(tile_hw[0]), int(tile_hw[1])
    image = tf.convert_to_tensor(image, tf.uint8)

    def one(g):
        x, y, w, h, left, top, cw, ch = (g[i] for i in range(8))
        crop = image[y : y + h, x : x + w]
        content = _fit(crop, ch, cw, resize)
        return tf.pad(
            content,
            [[top, th - top - ch], [left, tw - left - cw], [0, 0]],
            constant_values=pad,
        )

    return tf.map_fn(
        one,
        tf.convert_to_tensor(geometry, tf.int32),
        fn_output_signature=tf.TensorSpec([th, tw, 3], tf.uint8),
    )


def crop_targets(
    boxes: tf.Tensor,
    classes: tf.Tensor,
    valid: tf.Tensor,
    rect: tf.Tensor,
    tile_hw: Sequence[int],
    rules: TargetRules,
    content: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """§3.8 target rules on padded boxes, batched: the graph counterpart of
    ``tiling.crop_targets`` for a GPU mosaic.

    ``boxes [..., K, 4]`` are xyxy in source pixels, ``classes [..., K]``,
    ``valid [..., K]`` marks real rows, and ``rect [..., 4]`` is each
    window's ``(x, y, w, h)``; ``content [..., 4]`` places it on the tile
    (the whole tile when omitted). Returns tile-pixel boxes, classes and the
    surviving mask, compacted to the front in input order and capped at
    ``rules.max_boxes``. ``truncation="ignore"`` is not supported here.
    """
    if rules.truncation == "ignore":
        raise ValueError("crop_targets does not support truncation 'ignore'")
    th, tw = float(tile_hw[0]), float(tile_hw[1])
    boxes = tf.convert_to_tensor(boxes, tf.float32)
    rect = tf.cast(rect, tf.float32)
    x, y, w, h = (rect[..., i : i + 1] for i in range(4))
    if content is None:
        left = top = tf.zeros_like(x)
        cw, ch = tf.fill(tf.shape(x), tw), tf.fill(tf.shape(x), th)
    else:
        content = tf.cast(content, tf.float32)
        left, top, cw, ch = (content[..., i : i + 1] for i in range(4))
    x0 = tf.clip_by_value(boxes[..., 0], x, x + w)
    y0 = tf.clip_by_value(boxes[..., 1], y, y + h)
    x1 = tf.clip_by_value(boxes[..., 2], x, x + w)
    y1 = tf.clip_by_value(boxes[..., 3], y, y + h)
    area = tf.maximum(boxes[..., 2] - boxes[..., 0], 0.0) * tf.maximum(
        boxes[..., 3] - boxes[..., 1], 0.0
    )
    visible = tf.maximum(x1 - x0, 0.0) * tf.maximum(y1 - y0, 0.0)
    sx, sy = cw / w, ch / h
    out = tf.stack(
        [
            (x0 - x) * sx + left,
            (y0 - y) * sy + top,
            (x1 - x) * sx + left,
            (y1 - y) * sy + top,
        ],
        axis=-1,
    )
    keep = tf.logical_and(tf.cast(valid, tf.bool), visible > 0)
    if rules.truncation == "drop":
        frac = tf.math.divide_no_nan(visible, area)
        short = tf.minimum(
            out[..., 2] - out[..., 0], out[..., 3] - out[..., 1]
        )
        keep = keep & (frac >= rules.min_visibility)
        keep = keep & (short >= rules.min_box_px)
    rank = tf.cumsum(tf.cast(keep, tf.int32), axis=-1) - 1
    if rules.max_boxes is not None:
        keep = keep & (rank < rules.max_boxes)
    k = tf.shape(boxes)[-2]
    order = tf.argsort(
        tf.where(keep, tf.range(k), k + tf.range(k)), axis=-1, stable=True
    )
    gather = lambda t: tf.gather(t, order, batch_dims=order.shape.rank - 1)  # noqa: E731
    return (
        gather(out),
        gather(tf.convert_to_tensor(classes)),
        gather(keep),
    )


# --- validation -------------------------------------------------------------


def nms_multilabel(
    boxes: tf.Tensor,
    scores: tf.Tensor,
    iou: float,
    score_threshold: float = PROTOCOL_SCORE_FLOOR,
    max_det: int = 300,
    max_candidates: int = PROTOCOL_MAX_CANDIDATES,
    class_agnostic: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """§5 multi-label NMS on a batch, in fixed shapes.

    ``boxes [B, A, 4]`` xyxy and ``scores [B, A, C]``. Returns ``boxes [B,
    max_det, 4]``, ``scores [B, max_det]``, ``classes int32[B, max_det]``
    and ``count int32[B]``, rows past ``count`` zero. Matches the NumPy
    reference ``tiling.nms_multilabel`` up to the order of equal scores.
    """
    boxes = tf.convert_to_tensor(boxes, tf.float32)
    scores = tf.convert_to_tensor(scores, tf.float32)
    c = tf.shape(scores)[2]

    def one(args):
        b, s = args
        flat = tf.reshape(s, [-1])
        flat = tf.where(flat > score_threshold, flat, -1.0)
        k = tf.minimum(max_candidates, tf.size(flat))
        top, index = tf.math.top_k(flat, k=k)
        alive = top > score_threshold
        top, index = top[alive], index[alive]
        anchor, cls = index // c, index % c
        cand = tf.gather(b, anchor)
        # Suppress within each class on the boxes as given: a class offset
        # added to the coordinates loses float32 precision and changes IoUs.
        # Candidates are grouped by class in score order; each group keeps at
        # most max_det, then the kept candidates return to score order.
        group = tf.zeros_like(cls) if class_agnostic else cls
        order = tf.argsort(group, stable=True)
        _, _, counts = tf.unique_with_counts(tf.gather(group, order))
        starts = tf.cumsum(counts, exclusive=True)
        yxyx = tf.gather(tf.gather(cand, order), [1, 0, 3, 2], axis=1)
        ranked = tf.gather(top, order)

        def per_group(i, kept):
            lo, size = starts[i], counts[i]
            sel = tf.image.non_max_suppression(
                yxyx[lo : lo + size], ranked[lo : lo + size], max_det, iou
            )
            return i + 1, kept.write(i, tf.gather(order, sel + lo))

        # One extra empty element, as a TensorArray of none cannot concat.
        groups = tf.size(counts)
        kept = tf.TensorArray(
            tf.int32,
            size=groups + 1,
            infer_shape=False,
            element_shape=tf.TensorShape([None]),
        ).write(groups, tf.zeros([0], tf.int32))
        _, kept = tf.while_loop(
            lambda i, _: i < groups, per_group, (tf.constant(0), kept)
        )
        keep = tf.sort(kept.concat())[:max_det]
        n = tf.shape(keep)[0]
        pad = max_det - n
        return (
            tf.pad(tf.gather(cand, keep), [[0, pad], [0, 0]]),
            tf.pad(tf.gather(top, keep), [[0, pad]]),
            tf.pad(tf.gather(cls, keep), [[0, pad]]),
            n,
        )

    return tf.map_fn(
        one,
        (boxes, scores),
        fn_output_signature=(
            tf.TensorSpec([max_det, 4], tf.float32),
            tf.TensorSpec([max_det], tf.float32),
            tf.TensorSpec([max_det], tf.int32),
            tf.TensorSpec([], tf.int32),
        ),
    )


__all__ = [
    "KIND_CODES",
    "PAD_VALUE",
    "crop_targets",
    "decode_frame",
    "nms_multilabel",
    "render_windows",
    "visit_arrays",
    "visit_fn",
    "visit_windows",
]
