"""TensorFlow tiling ops against the NumPy core and OpenCV references."""

import struct

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
cv2 = pytest.importorskip("cv2")

from edgefirst.cameraadaptor.tensorflow import tiling as tft  # noqa: E402
from edgefirst.cameraadaptor.tiling import (  # noqa: E402
    Mix,
    ScaleConfig,
    TargetRules,
    TilingConfig,
    crop_targets,
    gt_extent,
    make_rng,
    nms_multilabel,
    pack_geometry,
    pack_placements,
    pack_targets,
    plan_epoch,
    sample_windows,
    tile_grid,
)
from edgefirst.cameraadaptor.tiling.pixels import render_window  # noqa: E402

T = (320, 320)


def _frame(h, w, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


# --- training ---------------------------------------------------------------


def test_visit_pipeline_draws_the_core_windows():
    cfg = TilingConfig(tile_size=T, frames_per_batch=2, seed=5)
    sizes = [(720, 1280), (900, 1600), (300, 500)]
    labels = [
        (np.array([[100, 100, 140, 150]], np.float32), np.array([2])),
        (np.zeros((0, 4), np.float32), np.zeros(0, np.int32)),
        (
            np.array([[10, 20, 60, 90], [200, 50, 260, 80]], np.float32),
            np.array([0, 1]),
        ),
    ]
    extents = [
        gt_extent(hw, b, T, cfg.grid_min_overlap)
        for hw, (b, _) in zip(sizes, labels, strict=True)
    ]
    epoch = 3
    plan = plan_epoch(extents, 6, cfg, epoch)
    frames, codes = tft.visit_arrays(plan)
    fn = tft.visit_fn(cfg, sizes, lambda i: labels[i], max_boxes=8)
    ds = tf.data.Dataset.from_tensor_slices((frames, codes)).map(
        lambda f, k: tft.visit_windows(
            fn, tf.constant(epoch, tf.int64), f, k, tf.constant(False), 8
        )
    )
    for visit, (geom, packed, _) in zip(plan.visits(), ds, strict=True):
        i = visit.frame
        windows = sample_windows(
            sizes[i],
            labels[i][0],
            labels[i][1],
            cfg,
            make_rng(cfg.seed, epoch, i, 0),
            kinds=visit.kinds,
        )
        np.testing.assert_array_equal(
            geom.numpy(), np.stack([pack_geometry(w) for w in windows])
        )
        np.testing.assert_array_equal(
            packed.numpy(),
            np.stack([pack_targets(w, w.out_hw, 8) for w in windows]),
        )


def test_native_windows_render_as_exact_crops():
    img = _frame(720, 1280)
    cfg = TilingConfig(
        tile_size=T,
        mix=Mix(1.0, 0.0, 0.0),
        grid_prob=0.0,
        scale=ScaleConfig(native_prob=1.0),
    )
    windows = sample_windows(
        (720, 1280),
        [[400, 300, 440, 340]],
        [0],
        cfg,
        np.random.default_rng(0),
        k=6,
    )
    geom = np.stack([pack_geometry(w) for w in windows])
    tiles = tft.render_windows(img, geom, T).numpy()
    for w, tile in zip(windows, tiles, strict=True):
        np.testing.assert_array_equal(
            tile, render_window(img, w.rect, w.out_hw, w.content)
        )


@pytest.mark.parametrize("resize", ["antialias", "bilinear"])
def test_resized_windows_match_opencv_within_rounding(resize):
    img = _frame(1080, 1920, seed=1)
    cfg = TilingConfig(
        tile_size=T, scale=ScaleConfig(native_prob=0.0), grid_prob=0.0
    )
    windows = sample_windows(
        (1080, 1920),
        [[900, 500, 960, 560]],
        [0],
        cfg,
        np.random.default_rng(2),
        k=24,
    )
    geom = np.stack([pack_geometry(w) for w in windows])
    tiles = tft.render_windows(img, geom, T, resize=resize).numpy()
    for w, tile in zip(windows, tiles, strict=True):
        ref = render_window(img, w.rect, w.out_hw, w.content, resize)
        diff = np.abs(tile.astype(int) - ref.astype(int))
        assert diff.mean() < 1.0, (w.kind, w.scale, diff.mean())
        assert np.percentile(diff, 99) <= 4


def test_letterboxed_grid_tiles_pad_with_114():
    img = _frame(200, 500)
    grid = tile_grid((200, 500), T, 0.1)
    tiles = tft.render_windows(img, pack_placements(grid), T).numpy()
    left, top, cw, ch = grid[0].content
    assert top > 0
    assert (tiles[0][:top] == 114).all()
    ref = render_window(img, grid[0].rect, T, grid[0].content)
    assert np.abs(tiles[0].astype(int) - ref.astype(int)).mean() < 1.0


def _exif_jpeg(img, value):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    tiff = b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", 1)
    tiff += struct.pack("<HHIHH", 0x0112, 3, 1, value, 0) + struct.pack(
        "<I", 0
    )
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    data = buf.tobytes()
    return data[:2] + app1 + data[2:]


TRANSFORMS = [
    lambda a: a,
    np.fliplr,
    lambda a: np.rot90(a, 2),
    np.flipud,
    lambda a: np.transpose(a, (1, 0, 2)),
    lambda a: np.rot90(a, -1),
    lambda a: np.rot90(a, 2).transpose(1, 0, 2),
    lambda a: np.rot90(a, 1),
]


@pytest.mark.parametrize("value", range(1, 9))
def test_decode_applies_exif_orientation_like_opencv(value, tmp_path):
    img = _frame(48, 80, seed=value)
    data = _exif_jpeg(img, value)
    path = tmp_path / "f.jpg"
    path.write_bytes(data)
    oriented = cv2.imread(str(path))
    stored = cv2.imread(
        str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    # The transform OpenCV applies, found exactly from its own decodes.
    (opencv,) = [t for t in TRANSFORMS if np.array_equal(t(stored), oriented)][
        :1
    ]
    plain = tf.io.decode_jpeg(tf.constant(data), channels=3).numpy()
    got = tft.decode_frame(tf.constant(data)).numpy()
    np.testing.assert_array_equal(got, opencv(plain))


def test_graph_target_rules_match_the_core():
    rng = np.random.default_rng(4)
    rules_list = [
        TargetRules(),
        TargetRules(truncation="keep"),
        TargetRules(max_boxes=5),
        TargetRules(min_visibility=0.5, min_box_px=4.0),
    ]
    k = 24
    for trial in range(40):
        rules = rules_list[trial % len(rules_list)]
        xy = rng.uniform(0, 900, (k, 2))
        boxes = np.concatenate([xy, xy + rng.uniform(1, 120, (k, 2))], 1)
        n = int(rng.integers(0, k))
        rect = (
            int(rng.integers(0, 400)),
            int(rng.integers(0, 400)),
            int(rng.integers(200, 640)),
            int(rng.integers(200, 640)),
        )
        content = None if trial % 3 else (0, 20, 320, 280)
        classes = rng.integers(0, 5, k)
        ref = crop_targets(
            rect,
            T,
            {"boxes": boxes[:n], "classes": classes[:n]},
            rules,
            content,
        )
        b, c, keep = tft.crop_targets(
            boxes.astype(np.float32),
            classes,
            np.arange(k) < n,
            np.asarray(rect),
            T,
            rules,
            None if content is None else np.asarray(content),
        )
        m = int(keep.numpy().sum())
        assert m == len(ref.boxes), trial
        assert keep.numpy()[:m].all()
        np.testing.assert_allclose(b.numpy()[:m], ref.boxes, atol=1e-3)
        np.testing.assert_array_equal(c.numpy()[:m], ref.classes)


def test_graph_target_rules_reject_ignore():
    with pytest.raises(ValueError, match="ignore"):
        tft.crop_targets(
            np.zeros((1, 4), np.float32),
            [0],
            [True],
            [0, 0, 10, 10],
            T,
            TargetRules(truncation="ignore"),
        )


# --- validation -------------------------------------------------------------


def _canon(b, s, c):
    k = np.lexsort((b[:, 1], b[:, 0], c, -s))
    return b[k], s[k], c[k]


@pytest.mark.parametrize("class_agnostic", [False, True])
def test_graph_nms_matches_the_reference(class_agnostic):
    rng = np.random.default_rng(6)
    batch, anchors, classes = 3, 800, 4
    xy = rng.uniform(0, 1200, (batch, anchors, 2))
    boxes = np.concatenate(
        [xy, xy + rng.uniform(4, 150, (batch, anchors, 2))], -1
    )
    scores = rng.random((batch, anchors, classes)) ** 6
    for iou, cap in [(0.7, 30000), (0.5, 200)]:
        b, s, c, n = tft.nms_multilabel(
            boxes,
            scores,
            iou,
            max_det=100,
            max_candidates=cap,
            class_agnostic=class_agnostic,
        )
        for i in range(batch):
            ref = nms_multilabel(
                boxes[i],
                scores[i],
                iou=iou,
                max_det=100,
                max_candidates=cap,
                class_agnostic=class_agnostic,
            )
            m = int(n[i])
            got = _canon(
                b.numpy()[i, :m],
                s.numpy()[i, :m],
                c.numpy()[i, :m].astype(np.int64),
            )
            want = _canon(*ref)
            assert m == len(want[0])
            np.testing.assert_allclose(got[0], want[0], atol=1e-3)
            np.testing.assert_allclose(got[1], want[1], rtol=1e-6)
            np.testing.assert_array_equal(got[2], want[2])


def test_graph_nms_keeps_boxes_at_exactly_the_threshold():
    boxes = np.array([[[0, 0, 10, 10], [0, 0, 10, 20]]], np.float32)
    scores = np.array([[[0.9], [0.8]]], np.float32)
    *_, n = tft.nms_multilabel(boxes, scores, 0.5)
    assert int(n[0]) == len(nms_multilabel(boxes[0], scores[0], 0.5)[0])
