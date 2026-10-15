import numpy as np
import pytest

torch = pytest.importorskip("torch")

from edgefirst.cameraadaptor.pytorch.tiling import (  # noqa: E402
    merge_frame,
    plan_frame,
    render_tiles,
    tile_targets,
)
from edgefirst.cameraadaptor.tiling import TargetRules  # noqa: E402


def test_render_tiles_are_native_crops_of_the_unpadded_frame():
    img = torch.rand(1, 3, 576, 960)
    plan = plan_frame((544, 960), (0, 16), 320, 0.1)
    tiles = render_tiles(img, plan)
    assert tiles.shape == (len(plan.tiles), 3, 320, 320)
    x, y = plan.tiles[1].origin
    torch.testing.assert_close(
        tiles[1], img[0, :, 16 + y : 16 + y + 320, x : x + 320]
    )


def test_small_frame_tile_is_letterboxed_with_pad_value():
    img = torch.rand(1, 3, 128, 160)
    plan = plan_frame((100, 160), (0, 14), 320, 0.1)
    (tile,) = render_tiles(img, plan)
    left, top, w, h = plan.tiles[0].content
    assert top > 0
    assert torch.all(tile[:, :top] == 114 / 255)


def test_small_frame_with_tile_aspect_fills_the_tile():
    img = torch.full((1, 3, 160, 160), 0.5)
    plan = plan_frame((160, 160), (0, 0), 320, 0.1)
    assert plan.tiles[0].content is None
    (tile,) = render_tiles(img, plan)
    assert tile.shape == (3, 320, 320)
    torch.testing.assert_close(tile, torch.full((3, 320, 320), 0.5))


def test_tile_targets_reject_ignore_truncation():
    plan = plan_frame((544, 960), (0, 0), 320, 0.1)
    boxes = np.array([[10, 10, 40, 40]], np.float32)
    classes = np.array([0])
    rules = TargetRules(truncation="ignore")
    with pytest.raises(ValueError, match="ignore"):
        tile_targets(boxes, classes, plan, rules)


def test_tile_targets_follow_the_grid():
    plan = plan_frame((544, 960), (0, 0), 320, 0.1)
    boxes = np.array([[10, 10, 40, 40]], np.float32)
    idx, cls, bboxes = tile_targets(boxes, np.array([0]), plan, TargetRules())
    assert idx.tolist() == [0.0]
    assert cls.shape == (1, 1)
    torch.testing.assert_close(
        bboxes, torch.tensor([[25 / 320, 25 / 320, 30 / 320, 30 / 320]])
    )


def test_merge_frame_lifts_merges_and_adds_pad():
    plan = plan_frame(
        (544, 960), (0, 16), 320, 0.1
    )  # tile x origins 0, 213, 427, 640
    empty = {
        "bboxes": torch.zeros((0, 4)),
        "conf": torch.zeros(0),
        "cls": torch.zeros(0),
    }
    per_tile = [empty] * len(plan.tiles)
    # Frame object [200, 100, 260, 140]: whole in tile 0, cut to [213, 260] by
    # tile 1 (IoS 1.0).
    per_tile[0] = {
        "bboxes": torch.tensor([[200.0, 100.0, 260.0, 140.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
    }
    per_tile[1] = {
        "bboxes": torch.tensor([[0.0, 100.0, 47.0, 140.0]]),
        "conf": torch.tensor([0.8]),
        "cls": torch.tensor([0.0]),
    }
    out = merge_frame(per_tile, plan)
    assert len(out["conf"]) == 1
    torch.testing.assert_close(
        out["bboxes"][0], torch.tensor([200.0, 116.0, 260.0, 156.0])
    )


def test_1080p_frame_fits_a_1088_high_tile():
    plan = plan_frame((1080, 1920), (0, 0), (1088, 1920), 0.1)
    (tile,) = plan.tiles
    assert tile.crop_size == (1920, 1080)
    assert tile.content == (
        0,
        4,
        1920,
        1080,
    )
