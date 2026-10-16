import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    Mix,
    ScaleConfig,
    TargetRules,
    TilingConfig,
    crop_targets,
    pack_geometry,
    pack_targets,
    sample_windows,
)

T = (640, 640)


def test_targets_pack_normalised_in_input_order_and_pad():
    r = crop_targets(
        (0, 0, 640, 640),
        T,
        {"boxes": [[64, 128, 192, 320], [0, 0, 32, 64]], "classes": [3, 1]},
        TargetRules(),
    )
    packed = pack_targets(r, T, 4)
    assert packed.shape == (4, 5)
    assert packed.dtype == np.float32
    np.testing.assert_allclose(
        packed[:2],
        [[0.2, 0.35, 0.2, 0.3, 3], [0.025, 0.05, 0.05, 0.1, 1]],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(packed[2:], -1.0)


def test_too_many_boxes_raise_instead_of_truncating():
    r = crop_targets(
        (0, 0, 640, 640),
        T,
        {
            "boxes": [[10 * i, 0, 10 * i + 8, 8] for i in range(3)],
            "classes": [0] * 3,
        },
        TargetRules(),
    )
    with pytest.raises(ValueError, match="max_boxes"):
        pack_targets(r, T, 2)


def test_ignore_regions_raise_instead_of_being_dropped():
    r = crop_targets(
        (0, 0, 640, 640),
        T,
        {"boxes": [[600, 0, 700, 100]], "classes": [0]},  # visibility 0.4
        TargetRules(truncation="ignore", min_visibility=0.5),
    )
    assert len(r.ignore_boxes) == 1
    with pytest.raises(ValueError, match="ignore"):
        pack_targets(r, T, 4)


def test_geometry_packs_rect_and_content():
    cfg = TilingConfig(
        mix=Mix(0.0, 0.0, 1.0), scale=ScaleConfig(native_prob=1.0)
    )
    (w,) = sample_windows(
        (360, 1280), [[10, 10, 50, 50]], [0], cfg, np.random.default_rng(0)
    )
    geom = pack_geometry(w)
    assert geom.dtype == np.int32
    assert tuple(geom[:4]) == w.rect
    assert tuple(geom[4:]) == w.content
    cfg = TilingConfig(
        mix=Mix(1.0, 0.0, 0.0),
        grid_prob=0.0,
        scale=ScaleConfig(native_prob=1.0),
    )
    (w,) = sample_windows(
        (2160, 3840), [[10, 10, 50, 50]], [0], cfg, np.random.default_rng(0)
    )
    assert w.content is None
    assert tuple(pack_geometry(w)[4:]) == (0, 0, 640, 640)


def test_window_targets_pack_directly():
    cfg = TilingConfig(mix=Mix(1.0, 0.0, 0.0), grid_prob=0.0)
    (w,) = sample_windows(
        (2160, 3840),
        [[100, 100, 140, 150]],
        [0],
        cfg,
        np.random.default_rng(1),
    )
    np.testing.assert_array_equal(
        pack_targets(w, w.out_hw, 8), pack_targets(w.targets, w.out_hw, 8)
    )
