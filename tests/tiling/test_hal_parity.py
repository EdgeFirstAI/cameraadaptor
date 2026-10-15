"""The Python grid, lift and merge reference against HAL itself (§6.3, §6.4).

Runs when HAL's Python bindings (``edgefirst-image``, ``edgefirst-decoder``)
are installed, as they are with the ``dev`` extra. The golden fixture pins
known cases; these compare live on randomised ones.
"""

import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    lift_tile_boxes,
    merge_tiled_detections,
    tile_grid,
)

image = pytest.importorskip("edgefirst.image")
decoder = pytest.importorskip("edgefirst.decoder")

TILES = (256, 320, 512, 640, 1024)
OVERLAPS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.333, 0.5)


def _frames(rng, n):
    for _ in range(n):
        t = int(rng.choice(TILES))
        yield (
            int(rng.integers(t // 2, 8 * t)),
            int(rng.integers(t // 2, 6 * t)),
            t,
            float(rng.choice(OVERLAPS)),
        )


def test_hal_version_is_recorded():
    assert image.version().startswith("0.35.")


@pytest.mark.parametrize("seed", range(4))
def test_grid_matches_hal(seed):
    rng = np.random.default_rng(seed)
    for w, h, t, r in _frames(rng, 50):
        ours = tile_grid((h, w), (t, t), r)
        hal = image.tile_grid(w, h, t, t, r)
        assert len(ours) == len(hal)
        for p, spec in zip(ours, hal, strict=True):
            src = spec.source
            assert p.rect == (src.x, src.y, src.width, src.height)


@pytest.mark.parametrize("seed", range(4))
def test_lift_matches_hal(seed):
    rng = np.random.default_rng(100 + seed)
    for w, h, t, r in _frames(rng, 25):
        grid = tile_grid((h, w), (t, t), r)
        for p in grid[:: max(1, len(grid) // 4)]:
            xy = rng.uniform(0.0, 0.7, (8, 2))
            boxes = np.concatenate(
                [xy, xy + rng.uniform(0.01, 0.3, (8, 2))], 1
            )
            boxes = boxes.astype(np.float32)
            placement = image.TilePlacement(
                p.index,
                len(grid),
                p.origin,
                p.crop_size,
                (w, h),
                p.letterbox,
            )
            hal, _, _ = decoder.lift_tile_boxes(
                boxes,
                np.ones(len(boxes), np.float32),
                np.zeros(len(boxes), np.uint64),
                placement,
            )
            np.testing.assert_allclose(
                lift_tile_boxes(boxes, p), hal, rtol=0, atol=1e-5
            )


def _detections(rng, n):
    xy = rng.uniform(0, 1500, (n, 2)).astype(np.float32)
    wh = rng.uniform(4, 120, (n, 2)).astype(np.float32)
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = np.round(rng.random(n), 3).astype(np.float32)
    classes = rng.integers(0, 3, n).astype(np.uint64)
    return boxes, scores, classes


@pytest.mark.parametrize("metric", ["ios", "iou"])
@pytest.mark.parametrize("mode", ["keep_best", "union"])
@pytest.mark.parametrize("class_agnostic", [False, True])
@pytest.mark.parametrize("max_det", [300, 20])
def test_merge_matches_hal(metric, mode, class_agnostic, max_det):
    rng = np.random.default_rng(7)
    cfg = decoder.MergeConfig(
        metric=getattr(decoder.MatchMetric, metric.capitalize()),
        threshold=0.5,
        class_agnostic=class_agnostic,
        max_det=max_det,
        mode=getattr(
            decoder.MergeMode, "KeepBest" if mode == "keep_best" else "Union"
        ),
    )
    for _ in range(20):
        boxes, scores, classes = _detections(rng, int(rng.integers(1, 400)))
        hb, hs, hc = decoder.merge_tiled_detections(
            boxes, scores, classes, cfg
        )
        ob, os_, oc = merge_tiled_detections(
            boxes,
            scores,
            classes.astype(np.int64),
            metric=metric,
            threshold=0.5,
            mode=mode,
            class_agnostic=class_agnostic,
            max_det=max_det,
        )
        np.testing.assert_array_equal(ob, hb)
        np.testing.assert_array_equal(os_, hs)
        np.testing.assert_array_equal(oc, hc.astype(np.int64))
