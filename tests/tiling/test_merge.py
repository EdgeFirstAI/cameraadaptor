import json
import pathlib

import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    TilePlacement,
    lift_tile_boxes,
    merge_tiled_detections,
)

GOLDEN = json.loads(
    (
        pathlib.Path(__file__).parents[1] / "fixtures" / "tiling_golden.json"
    ).read_text()
)


def _placement(origin, crop, letterbox, tile=640):
    content = None
    if letterbox is not None:
        l0, t0, l1, t1 = letterbox
        content = (
            round(l0 * tile),
            round(t0 * tile),
            round((l1 - l0) * tile),
            round((t1 - t0) * tile),
        )
    return TilePlacement(0, tuple(origin), tuple(crop), (tile, tile), content)


@pytest.mark.parametrize("case", GOLDEN["lift"])
def test_lift_matches_hal(case):
    out = lift_tile_boxes(
        case["boxes"],
        _placement(case["origin"], case["crop"], case["letterbox"]),
    )
    assert out.dtype == np.float32
    np.testing.assert_allclose(
        out, case["expect"], atol=case["tol"] or 0, rtol=0
    )


@pytest.mark.parametrize("case", GOLDEN["merge"], ids=lambda c: c["name"])
def test_merge_matches_hal(case):
    boxes, scores, _ = merge_tiled_detections(
        case["boxes"],
        case["scores"],
        case["classes"],
        metric=case["metric"],
        mode=case["mode"],
        class_agnostic=case["class_agnostic"],
    )
    np.testing.assert_array_equal(
        boxes, np.asarray(case["expect_boxes"], np.float32)
    )
    np.testing.assert_array_equal(
        scores, np.asarray(case["expect_scores"], np.float32)
    )


def test_merge_ties_keep_input_order_and_truncate():
    boxes = [[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]]
    b, s, c = merge_tiled_detections(
        boxes, [0.5, 0.5, 0.9], [0, 1, 2], max_det=2
    )
    np.testing.assert_array_equal(c, [2, 0])


def test_merge_nested_same_class_is_suppressed_by_ios():
    b, _, _ = merge_tiled_detections(
        [[0, 0, 100, 100], [10, 10, 20, 20]], [0.9, 0.8], [0, 0]
    )
    assert len(b) == 1


def test_merge_empty():
    b, s, c = merge_tiled_detections(
        np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    )
    assert b.shape == (0, 4)
    assert s.shape == (0,)
    assert c.shape == (0,)


@pytest.mark.parametrize("case", GOLDEN["hal_random"] or [None])
def test_hal_generated_grids(case):
    if case is None:
        pytest.skip(
            "run tests/tiling/gen_golden_from_hal.py with HAL installed to "
            "add random cases"
        )
    from edgefirst.cameraadaptor.tiling import axis_origins

    assert axis_origins(case["L"], case["t"], case["r"]) == case["origins"]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("score_threshold", [0.0, 0.3])
def test_merge_early_stop_matches_merging_everything(seed, score_threshold):
    rng = np.random.default_rng(seed)
    n = 800
    xy = rng.uniform(0, 400, (n, 2)).astype(np.float32)
    wh = rng.uniform(4, 60, (n, 2)).astype(np.float32)
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    scores[rng.choice(n, 40, replace=False)] = 0.5  # ties
    classes = rng.integers(0, 3, n)
    full = merge_tiled_detections(
        boxes, scores, classes, max_det=10**9, score_threshold=score_threshold
    )
    for k in (0, 1, 50, 300):
        capped = merge_tiled_detections(
            boxes, scores, classes, max_det=k, score_threshold=score_threshold
        )
        for a, b in zip(capped, full, strict=True):
            np.testing.assert_array_equal(a, b[:k])


def test_merge_below_score_threshold_is_empty():
    b, s, c = merge_tiled_detections(
        np.array([[0, 0, 10, 10]], np.float32),
        np.array([0.1], np.float32),
        np.array([0]),
        score_threshold=0.5,
    )
    assert b.shape == (0, 4)
    assert len(s) == 0
    assert len(c) == 0


def test_frame_merge_lifts_merges_and_clips_to_the_frame():
    from edgefirst.cameraadaptor.tiling import (
        merge_frame_detections,
        pack_placements,
        tile_grid,
    )

    grid = tile_grid((300, 1000), (320, 320), 0.1)
    rng = np.random.default_rng(3)
    per_tile = []
    for _ in grid:
        xy = rng.uniform(0, 0.9, (6, 2)).astype(np.float32)
        boxes = np.concatenate([xy, xy + 0.3], axis=1)  # some past the edge
        per_tile.append((boxes, rng.random(6), rng.integers(0, 2, 6)))
    b, s, c = merge_frame_detections(per_tile, grid, (300, 1000))
    assert len(b)
    assert (b[:, [0, 2]] <= 1000).all()
    assert (b[:, [1, 3]] <= 300).all()
    assert (np.diff(s) <= 0).all()
    geom = pack_placements(grid)
    assert geom.shape == (len(grid), 8)
    assert tuple(geom[0, :4]) == grid[0].rect
    torch = pytest.importorskip("torch")
    from edgefirst.cameraadaptor.pytorch.tiling import merge_frame, plan_frame

    plan = plan_frame((300, 1000), (0, 0), 320, 0.1)
    preds = [
        {
            "bboxes": torch.from_numpy(t[0] * 320),
            "conf": torch.from_numpy(np.asarray(t[1], np.float32)),
            "cls": torch.from_numpy(np.asarray(t[2])),
        }
        for t in per_tile
    ]
    out = merge_frame(preds, plan)
    np.testing.assert_allclose(out["bboxes"].numpy(), b, atol=1e-4)
    np.testing.assert_array_equal(out["cls"].numpy(), c)
