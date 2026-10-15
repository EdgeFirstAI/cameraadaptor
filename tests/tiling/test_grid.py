import json
import pathlib

import pytest

from edgefirst.cameraadaptor.tiling import (
    axis_origins,
    fit_content,
    letterbox_rect,
    tile_grid,
)

GOLDEN = json.loads(
    (
        pathlib.Path(__file__).parents[1] / "fixtures" / "tiling_golden.json"
    ).read_text()
)


@pytest.mark.parametrize(
    "case", GOLDEN["evendist"], ids=lambda c: f"{c['L']}-{c['t']}-{c['r']}"
)
def test_evendist_matches_hal(case):
    assert axis_origins(case["L"], case["t"], case["r"]) == case["origins"]


@pytest.mark.parametrize("case", GOLDEN["grid_counts"])
def test_grid_counts(case):
    t = case["tile"]
    assert (
        len(tile_grid(tuple(case["frame_hw"]), (t, t), case["r"]))
        == case["count"]
    )


@pytest.mark.parametrize("case", GOLDEN["letterbox"])
def test_letterbox_matches_hal(case):
    assert list(letterbox_rect(*case["src"], *case["dst"])) == case["rect"]


def test_grid_is_row_major_and_inside_frame():
    grid = tile_grid((1080, 1920), (640, 640), 0.1)
    assert [p.index for p in grid] == list(range(len(grid)))
    assert [p.origin for p in grid[:4]] == [
        (0, 0),
        (427, 0),
        (853, 0),
        (1280, 0),
    ]
    for p in grid:
        x, y, w, h = p.rect
        assert x >= 0 and x + w <= 1920 and y >= 0 and y + h <= 1080
        assert p.content is None and p.letterbox is None


def test_small_frame_single_letterboxed_tile():
    (p,) = tile_grid((300, 400), (640, 640), 0.1)
    assert p.rect == (0, 0, 400, 300)
    assert p.content == (0, 80, 640, 480)
    assert p.letterbox == pytest.approx((0.0, 0.125, 1.0, 0.875))


def test_fit_content_none_when_crop_fills_tile():
    assert fit_content(640, 640, (640, 640)) is None
    assert fit_content(832, 832, (640, 640)) is None
    assert fit_content(400, 300, (640, 640), fit="stretch") is None


def test_frames_per_batch_fits_the_largest_grid_in_the_budget():
    from edgefirst.cameraadaptor.tiling import frames_per_batch

    # With 10% minimum overlap, 960x544 at 320 tiles is a 4x2 grid and 640x360
    # is 3x2
    assert frames_per_batch([(544, 960), (360, 640)], (320, 320), 0.1, 64) == 8
    assert frames_per_batch([(360, 640)], (320, 320), 0.1, 64) == 10
    assert frames_per_batch([(544, 960)], (320, 320), 0.1, 3) == 1
