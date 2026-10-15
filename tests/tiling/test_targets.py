import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import TargetRules, crop_targets

T = (640, 640)


def _crop(
    boxes,
    rect=(50, 0, 640, 640),
    rules=None,
    content=None,
    classes=None,
):
    rules = rules if rules is not None else TargetRules()
    classes = classes if classes is not None else list(range(len(boxes)))
    return crop_targets(
        rect,
        T,
        {"boxes": np.asarray(boxes, float), "classes": classes},
        rules,
        content,
    )


def test_visibility_edge_exactly_quarter_is_kept():
    r = _crop([[20, 100, 60, 140]])  # 10 of 40 px visible on x -> 0.25
    assert len(r.boxes) == 1 and r.clipped.tolist() == [True]
    np.testing.assert_allclose(r.boxes[0], [0, 100, 10, 140])


def test_visibility_just_below_quarter_is_dropped():
    r = _crop([[18, 100, 60.5, 140]])
    assert len(r.boxes) == 0 and r.dropped_visibility == 1


def test_pixel_floor_edges():
    r = _crop([[100, 100, 102, 140], [200, 100, 201.9, 140]])
    assert r.classes.tolist() == [0] and r.dropped_min_px == 1


def test_scale_and_letterbox_transform():
    r = crop_targets(
        (0, 0, 400, 300),
        T,
        {"boxes": [[100, 100, 200, 150]], "classes": [3]},
        TargetRules(),
        content=(0, 80, 640, 480),
    )
    np.testing.assert_allclose(r.boxes, [[160, 240, 320, 320]])
    assert r.classes.tolist() == [3]


def test_downscaled_window_maps_to_tile_pixels():
    r = crop_targets(
        (0, 0, 1280, 1280),
        T,
        {"boxes": [[640, 640, 680, 680]], "classes": [0]},
        TargetRules(),
    )
    np.testing.assert_allclose(r.boxes, [[320, 320, 340, 340]])


def test_keep_mode_keeps_any_visible():
    r = _crop(
        [[20, 100, 60.5, 140], [100, 100, 101, 101]],
        rules=TargetRules(truncation="keep"),
    )
    assert len(r.boxes) == 2


def test_ignore_mode_returns_ignore_band():
    r = _crop(
        [[18, 100, 60.5, 140], [45, 100, 145, 140]],
        rules=TargetRules(truncation="ignore"),
    )
    assert len(r.boxes) == 1  # second box: 95/100 visible
    assert r.ignore_boxes.shape == (1, 4)  # first box: ~0.247 visible, >= 0.05


def test_outside_boxes_are_not_counted_as_dropped():
    r = _crop([[1000, 1000, 1010, 1010]])
    assert (
        len(r.boxes) == 0
        and r.dropped_visibility == 0
        and r.dropped_min_px == 0
    )


def test_cap_keeps_input_order():
    r = _crop(
        [[100 + 20 * i, 100, 110 + 20 * i, 110] for i in range(5)],
        rules=TargetRules(max_boxes=3),
    )
    assert r.classes.tolist() == [0, 1, 2] and r.dropped_cap == 2


def test_reserved_target_kinds_raise():
    with pytest.raises(NotImplementedError, match="segments"):
        crop_targets(
            (0, 0, 640, 640),
            T,
            {"boxes": [], "classes": [], "segments": []},
            TargetRules(),
        )
