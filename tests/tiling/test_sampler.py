import math

import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    Mix,
    ScaleConfig,
    TilingConfig,
    make_rng,
    repeat_factors,
    sample_windows,
)

CHI2_63_P01 = 92.01  # chi-square critical value, 63 dof, p = 0.01
T = 640


def _cfg(**kw):
    base = {
        "tile_size": (T, T),
        "mix": Mix(1.0, 0.0, 0.0),
        "grid_prob": 0.0,
        "cut_prob": 0.0,
        "scale": ScaleConfig(native_prob=1.0),
    }
    base.update(kw)
    return TilingConfig(**base)


def _hits(rect, boxes):
    x, y, w, h = rect
    b = np.asarray(boxes, float).reshape(-1, 4)
    return (np.minimum(b[:, 2], x + w) - np.maximum(b[:, 0], x) > 0) & (
        np.minimum(b[:, 3], y + h) - np.maximum(b[:, 1], y) > 0
    )


def test_determinism():
    boxes, cls = [[1000, 500, 1030, 530]], [0]
    a = sample_windows(
        (2160, 3840), boxes, cls, TilingConfig(), make_rng(1, 2, 3, 0), k=8
    )
    b = sample_windows(
        (2160, 3840), boxes, cls, TilingConfig(), make_rng(1, 2, 3, 0), k=8
    )
    assert [(w.kind, w.rect) for w in a] == [(w.kind, w.rect) for w in b]


def test_k_zero_returns_none():
    assert (
        sample_windows(
            (2160, 3840),
            [[1000, 500, 1030, 530]],
            [0],
            TilingConfig(),
            np.random.default_rng(13),
            k=0,
        )
        == []
    )


def test_containment_over_random_frames():
    rng = np.random.default_rng(0)
    for i in range(300):
        fh, fw = int(rng.integers(200, 2500)), int(rng.integers(200, 4000))
        n = int(rng.integers(0, 12))
        xy = rng.uniform(0, [fw - 5, fh - 5], size=(n, 2))
        wh = rng.uniform(4, 200, size=(n, 2))
        boxes = np.concatenate([xy, np.minimum(xy + wh, [fw, fh])], 1)
        for w in sample_windows(
            (fh, fw),
            boxes,
            np.zeros(n),
            TilingConfig(placement_candidates=1),
            make_rng(0, 0, i, 0),
            k=4,
        ):
            x, y, ww, hh = w.rect
            assert x >= 0 and y >= 0 and x + ww <= fw and y + hh <= fh
            assert np.all(w.boxes >= -1e-4) and np.all(w.boxes <= T + 1e-4)


def test_no_centre_bias_and_centred_sampler_fails():
    b = [1910, 1070, 1930, 1090]  # 20 px object far from every edge
    cfg, rng = _cfg(placement_candidates=1), np.random.default_rng(1)
    centres = np.array(
        [
            w.anchor_center
            for w in sample_windows((2160, 3840), [b], [0], cfg, rng, k=10_000)
        ]
    )
    feasible = (centres * T - 10) / (
        T - 20
    )  # centre range [10, 630] px -> [0, 1]
    hist, _, _ = np.histogram2d(
        feasible[:, 0], feasible[:, 1], bins=8, range=[[0, 1], [0, 1]]
    )
    expected = hist.sum() / 64
    assert ((hist - expected) ** 2 / expected).sum() < CHI2_63_P01
    centred = np.full((10_000, 2), 0.5)  # the v2 sampler centres every anchor
    hist_v2, _, _ = np.histogram2d(
        centred[:, 0], centred[:, 1], bins=8, range=[[0, 1], [0, 1]]
    )
    assert ((hist_v2 - expected) ** 2 / expected).sum() > CHI2_63_P01


def test_no_border_pile_up():
    b = [100, 1070, 120, 1090]  # feasible x origins 0..100 (101 values)
    windows = sample_windows(
        (2160, 3840),
        [b],
        [0],
        _cfg(placement_candidates=1),
        np.random.default_rng(2),
        k=10_000,
    )
    counts = np.bincount([w.rect[0] for w in windows])
    assert counts.max() <= 2 / 101 * 10_000


def test_background_windows_never_touch_ground_truth():
    rng = np.random.default_rng(3)
    boxes = np.array(
        [
            [x, y, x + 60, y + 60]
            for x in range(0, 3800, 1200)
            for y in range(0, 2100, 1000)
        ],
        float,
    )
    cfg = TilingConfig(mix=Mix(0.0, 1.0, 0.0), grid_prob=0.5)
    windows = sample_windows(
        (2160, 3840), boxes, np.zeros(len(boxes)), cfg, rng, k=2000
    )
    background = [w for w in windows if w.kind == "background"]
    assert len(background) > 1000
    for w in background:
        assert not _hits(w.rect, boxes).any()


def test_mixture_shares_and_empty_frame_redistribution():
    kinds = [
        w.kind
        for w in sample_windows(
            (2160, 3840),
            [[1000, 500, 1030, 530]],
            [0],
            TilingConfig(),
            np.random.default_rng(4),
            k=10_000,
        )
    ]
    for kind, share in (
        ("positive", 0.75),
        ("background", 0.15),
        ("full_frame", 0.10),
    ):
        assert abs(kinds.count(kind) / 1e4 - share) < 0.02
    empty = [
        w.kind
        for w in sample_windows(
            (2160, 3840),
            np.zeros((0, 4)),
            [],
            TilingConfig(),
            np.random.default_rng(5),
            k=10_000,
        )
    ]
    assert empty.count("positive") == 0
    assert abs(empty.count("background") / 1e4 - 0.90) < 0.02


def test_cut_extension_keeps_quarter_visibility():
    b = np.array([1000, 1000, 1100, 1100], float)
    vis = []
    for w in sample_windows(
        (2160, 3840),
        [b],
        [0],
        _cfg(cut_prob=1.0),
        np.random.default_rng(6),
        k=2000,
    ):
        x, y, ww, hh = w.rect
        iw = min(b[2], x + ww) - max(b[0], x)
        ih = min(b[3], y + hh) - max(b[1], y)
        vis.append(iw * ih / 10_000)
        assert len(w.boxes) == 1
    assert min(vis) >= 0.25 and min(vis) < 1.0


def test_scale_rule_and_final_phase():
    cfg = TilingConfig(mix=Mix(1.0, 0.0, 0.0), grid_prob=0.0)
    windows = sample_windows(
        (2160, 3840),
        [[1900, 1000, 1940, 1040]],
        [0],
        cfg,
        np.random.default_rng(7),
        k=10_000,
    )
    native = np.array([w.rect[2] == T for w in windows])
    assert abs(native.mean() - 0.5) < 0.02
    log_s = np.log([w.scale for w in windows if w.rect[2] != T])
    assert abs(log_s.mean()) < 0.02
    final = sample_windows(
        (2160, 3840),
        [[1900, 1000, 1940, 1040]],
        [0],
        TilingConfig(),
        np.random.default_rng(8),
        k=2000,
        phase="final",
    )
    assert all(w.kind != "full_frame" and w.rect[2] == T for w in final)


def test_small_frame_gives_single_letterboxed_window():
    for w in sample_windows(
        (300, 400),
        [[100, 100, 200, 150]],
        [0],
        TilingConfig(),
        np.random.default_rng(9),
        k=20,
    ):
        assert w.rect == (0, 0, 400, 300) and w.content == (0, 80, 640, 480)
        np.testing.assert_allclose(w.boxes, [[160, 240, 320, 320]])


def test_dense_frame_falls_back_without_hanging():
    boxes = np.array(
        [
            [x, y, x + 100, y + 100]
            for x in range(0, 1000, 100)
            for y in range(0, 1000, 100)
        ],
        float,
    )
    cfg = TilingConfig(mix=Mix(0.0, 1.0, 0.0))
    windows = sample_windows(
        (1000, 1000),
        boxes,
        np.zeros(len(boxes)),
        cfg,
        np.random.default_rng(10),
        k=50,
    )
    assert all(w.kind == "positive" and w.fallback for w in windows)


def test_hard_regions_place_background_windows():
    cfg = TilingConfig(
        mix=Mix(0.0, 1.0, 0.0),
        grid_prob=0.0,
        hard_fraction=1.0,
        scale=ScaleConfig(native_prob=1.0),
    )
    hard = np.array([[3000, 1500, 3050, 1550]], np.float32)
    for w in sample_windows(
        (2160, 3840),
        [[100, 100, 130, 130]],
        [0],
        cfg,
        np.random.default_rng(11),
        k=200,
        hard_regions=hard,
    ):
        x, y, ww, hh = w.rect
        assert x <= 3000 and x + ww >= 3050 and y <= 1500 and y + hh >= 1550


def test_grid_snapping_uses_deployment_grid():
    cfg = _cfg(grid_prob=1.0, grid_min_overlap=0.1)
    for w in sample_windows(
        (1080, 1920),
        [[500, 500, 530, 530]],
        [0],
        cfg,
        np.random.default_rng(12),
        k=100,
    ):
        assert w.grid_index is not None and w.rect[0] in (0, 427, 853, 1280)


def test_class_balanced_anchors_favour_rare_classes():
    boxes = np.array(
        [[180 * i, 100, 180 * i + 20, 120] for i in range(10)], float
    )
    classes = np.array([0] * 9 + [1])
    rf = repeat_factors(
        [np.array([0])] * 999 + [np.array([1])], 2, t=0.1
    )  # rare class weight 10

    def rare_rate(cfg):
        firsts = [
            sample_windows(
                (1080, 1920),
                boxes,
                classes,
                cfg,
                np.random.default_rng(s),
                k=1,
                class_repeat=rf,
            )[0]
            for s in range(2000)
        ]
        return np.mean([1 in w.classes.tolist() for w in firsts])

    assert (
        rare_rate(_cfg(anchor_weighting="class_balanced"))
        > rare_rate(_cfg()) + 0.15
    )


def test_class_balanced_requires_repeat_factors():
    with pytest.raises(ValueError, match="class_repeat"):
        sample_windows(
            (1080, 1920),
            [[0, 0, 10, 10]],
            [0],
            _cfg(anchor_weighting="class_balanced"),
            np.random.default_rng(0),
            k=1,
        )


def test_repeat_factors():
    rf = repeat_factors(
        [np.array([0]), np.array([0, 1]), np.array([0])], 3, t=0.5
    )
    assert (
        rf[0] == 1.0
        and rf[1] == pytest.approx(math.sqrt(0.5 / (1 / 3)))
        and rf[2] == 1.0
    )


def test_explicit_kinds_are_drawn_in_order():
    boxes, cls = [[1000, 500, 1030, 530], [2000, 900, 2040, 940]], [0, 0]
    kinds = ["positive", "background", "full_frame", "positive"]
    got = sample_windows(
        (2160, 3840),
        boxes,
        cls,
        TilingConfig(),
        make_rng(0, 0, 0, 0),
        kinds=kinds,
    )
    assert [w.kind for w in got] == kinds


def test_explicit_kinds_reject_positive_without_ground_truth():
    with pytest.raises(ValueError, match="ground truth"):
        sample_windows(
            (2160, 3840),
            [],
            [],
            TilingConfig(),
            make_rng(0, 0, 0, 0),
            kinds=["positive"],
        )
    got = sample_windows(
        (2160, 3840),
        [],
        [],
        TilingConfig(),
        make_rng(0, 0, 0, 0),
        kinds=["background"] * 3,
    )
    assert all(w.kind == "background" and len(w.boxes) == 0 for w in got)


def test_explicit_kinds_background_falls_back_on_dense_frame():
    xs = np.arange(0, 3840, 40)
    ys = np.arange(0, 2160, 40)
    boxes = [[x, y, x + 30, y + 30] for x in xs for y in ys]
    got = sample_windows(
        (2160, 3840),
        boxes,
        [0] * len(boxes),
        _cfg(),
        make_rng(0, 0, 0, 0),
        kinds=["background"] * 5,
    )
    assert all(w.kind == "positive" and w.fallback for w in got)


def test_default_k_is_one():
    got = sample_windows(
        (2160, 3840),
        [[1000, 500, 1030, 530]],
        [0],
        TilingConfig(),
        make_rng(0, 0, 0, 0),
    )
    assert len(got) == 1


def _cluster_frame(rng):
    lone = [[200, 200, 230, 230]]
    cluster = [
        [2000 + 60 * i, 1000 + 60 * j, 2030 + 60 * i, 1030 + 60 * j]
        for i in range(4)
        for j in range(4)
    ]
    return lone + cluster


def test_placement_candidates_favour_tiles_with_more_objects():
    boxes = _cluster_frame(None)
    cls = [0] * len(boxes)

    def mean_objects(n):
        cfg = _cfg(placement_candidates=n, cut_prob=0.0, grid_prob=0.0)
        counts = [
            len(w.boxes)
            for i in range(300)
            for w in sample_windows(
                (2160, 3840),
                boxes,
                cls,
                cfg,
                make_rng(0, 0, i, 0),
                kinds=["positive"] * 4,
            )
        ]
        return float(np.mean(counts))

    assert mean_objects(4) > mean_objects(1) + 0.5


def _ks_uniform(u):
    """Kolmogorov-Smirnov statistic of ``u`` against Uniform(0, 1)."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    i = np.arange(1, n + 1)
    return max(np.max(i / n - u), np.max(u - (i - 1) / n))


def test_placement_candidates_keep_anchor_uniform_on_single_object_frames():
    box = [1800, 1000, 1830, 1030]
    cfg = _cfg(placement_candidates=4, cut_prob=0.0, grid_prob=0.0)
    u = []
    for i in range(2000):
        (w,) = sample_windows(
            (2160, 3840),
            [box],
            [0],
            cfg,
            make_rng(0, 0, i, 0),
            kinds=["positive"],
        )
        u.append((box[0] - w.rect[0]) / (w.rect[2] - (box[2] - box[0])))
    # Asymptotic critical value at the 1% level.
    assert _ks_uniform(u) < 1.628 / np.sqrt(len(u))


def test_one_candidate_reproduces_single_draw():
    boxes = _cluster_frame(None)
    cls = [0] * len(boxes)
    cfg = _cfg(placement_candidates=1, grid_prob=0.25, cut_prob=0.1)
    a = sample_windows(
        (2160, 3840),
        boxes,
        cls,
        cfg,
        make_rng(5, 0, 0, 0),
        kinds=["positive"] * 6,
    )
    b = sample_windows(
        (2160, 3840), boxes, cls, cfg, make_rng(5, 0, 0, 0), k=6
    )
    assert [w.rect for w in a] == [w.rect for w in b]


def test_placement_candidates_keep_cut_extension_rate():
    box = [1800, 1000, 1864, 1064]

    def cut_rate(n):
        cfg = _cfg(placement_candidates=n, cut_prob=1.0, grid_prob=0.0)
        cut = 0
        for i in range(2000):
            (w,) = sample_windows(
                (2160, 3840),
                [box],
                [0],
                cfg,
                make_rng(0, 0, i, 0),
                kinds=["positive"],
            )
            x, y, cw, ch = w.rect
            cut += not (
                box[0] >= x
                and box[1] >= y
                and box[2] <= x + cw
                and box[3] <= y + ch
            )
        return cut / 2000

    one, four = cut_rate(1), cut_rate(4)
    assert one > 0.05
    assert abs(four - one) <= 0.03
