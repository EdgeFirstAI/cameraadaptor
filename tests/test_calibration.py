"""Model-free calibration selection and tiled sample planning (§6.5)."""

import numpy as np
import pytest

from edgefirst.cameraadaptor.calibration import (
    HIST_BINS,
    extremum_positions,
    image_descriptor,
    plan_tile_samples,
    pool_descriptors,
    select_max_dynamic_range,
)
from edgefirst.cameraadaptor.tiling import tile_grid


def _pool(images):
    return pool_descriptors(
        [None if im is None else image_descriptor(im) for im in images]
    )


def _synthetic(n_mid=10, seed=7):
    """Index 0 all black (global minimum), index 1 all white (maximum)."""
    rng = np.random.default_rng(seed)
    images = [
        np.zeros((32, 32, 3), np.uint8),
        np.full((32, 32, 3), 255, np.uint8),
    ]
    images += [
        rng.integers(60, 200, (32, 32, 3), dtype=np.uint8)
        for _ in range(n_mid)
    ]
    return images


def test_extrema_frames_are_always_selected():
    pool = _pool(_synthetic())
    selected = select_max_dynamic_range(pool, count=5)
    assert 0 in selected
    assert 1 in selected
    sel, valid = np.array(selected), pool["valid"]
    for c in range(pool["channels"]):
        assert pool["mins"][sel, c].min() == pool["mins"][valid, c].min()
        assert pool["maxs"][sel, c].max() == pool["maxs"][valid, c].max()


def test_selection_is_deterministic():
    images = _synthetic(n_mid=20)
    runs = [select_max_dynamic_range(_pool(images), 8, seed=42) for _ in "ab"]
    assert runs[0] == runs[1]


def test_count_is_clamped_to_the_pool():
    pool = _pool(_synthetic(n_mid=3))
    assert select_max_dynamic_range(pool, count=500) == list(range(5))


def test_saturated_coverage_fills_by_seeded_draw():
    pool = _pool([np.full((16, 16, 3), 128, np.uint8) for _ in range(30)])
    a = select_max_dynamic_range(pool, count=10, seed=42)
    assert a == select_max_dynamic_range(pool, count=10, seed=42)
    assert len(a) == 10
    assert a == sorted(a)


def test_undecodable_frames_keep_their_index_and_are_never_selected():
    images = _synthetic(n_mid=4)
    images.insert(3, None)
    pool = _pool(images)
    assert not pool["valid"][3]
    assert pool["num_images"] == len(images)
    assert 3 not in select_max_dynamic_range(pool, count=len(images))


def test_channel_mismatch_is_masked():
    images = _synthetic(n_mid=2)
    images.append(np.zeros((8, 8, 1), np.uint8))
    assert not _pool(images)["valid"][-1]


def test_empty_or_undecodable_pools_raise():
    with pytest.raises(ValueError, match="empty"):
        pool_descriptors([])
    with pytest.raises(ValueError, match="no decodable"):
        pool_descriptors([None, None])


def test_negative_counts_are_rejected_and_zero_selects_nothing():
    pool = _pool(_synthetic())
    with pytest.raises(ValueError, match="count"):
        select_max_dynamic_range(pool, count=-1)
    assert select_max_dynamic_range(pool, count=0) == []


def test_plan_follows_pool_order_whatever_the_selected_order():
    sizes = [(1080, 1920), (1080, 1920)]
    forward = plan_tile_samples(sizes, [0, 1], T, 0.1, count=3)
    assert plan_tile_samples(sizes, [1, 0], T, 0.1, count=3) == forward
    assert plan_tile_samples(sizes, [1, 0, 1], T, 0.1, count=3) == forward
    assert [s.index for s in forward] == [0, 0, 1]


def test_descriptor_rejects_non_uint8_frames():
    with pytest.raises(ValueError, match="uint8"):
        image_descriptor(np.zeros((4, 4, 3), np.float32))


def test_histograms_are_normalised():
    pool = _pool(_synthetic(n_mid=2))
    sums = pool["hists"][pool["valid"]].sum(axis=2)
    np.testing.assert_allclose(sums, 1.0)
    assert pool["hists"].shape[2] == HIST_BINS


def test_descriptor_records_the_first_extremum_positions():
    img = np.full((4, 5, 3), 100, np.uint8)
    img[1, 2, 0] = 3  # first minimum of channel 0
    img[3, 0, 0] = 3
    img[2, 4, 1] = 250  # maximum of channel 1
    d = image_descriptor(img)
    assert tuple(d.min_pos[0]) == (1, 2)
    assert tuple(d.max_pos[1]) == (2, 4)


def test_extremum_positions_follow_the_seeded_frames():
    images = []
    for lo, hi in [(50, 200), (0, 200), (50, 255)]:
        img = np.full((8, 8, 3), 100, np.uint8)
        img[2, 3] = lo
        img[5, 6] = hi
        images.append(img)
    pool = _pool(images)
    positions = extremum_positions(pool, select_max_dynamic_range(pool, 3))
    assert set(positions) == {1, 2}
    assert set(positions[1]) == {(2, 3)}
    assert set(positions[2]) == {(5, 6)}
    assert extremum_positions(pool, [0]) == {}


T = (320, 320)


def test_extremum_frames_contribute_the_tiles_holding_the_pixels():
    grid = tile_grid((544, 960), T, 0.1)
    samples = plan_tile_samples(
        [(544, 960)],
        [0],
        T,
        0.1,
        extrema={0: [(10, 10), (12, 12), (500, 900)]},
    )
    holding = next(
        p.index
        for p in grid
        if p.origin[0] <= 900 < p.origin[0] + 320
        and p.origin[1] <= 500 < p.origin[1] + 320
    )
    assert [s.placement.index for s in samples] == sorted({0, holding})


def test_other_frames_get_one_seeded_tile_and_full_frame_adds_the_frame():
    sizes = [(544, 960), (200, 300)]
    samples = plan_tile_samples(sizes, [0, 1], T, 0.1, full_frame=True)
    assert [(s.index, s.placement is None) for s in samples] == [
        (0, False),
        (0, True),
        (1, False),
        (1, True),
    ]
    assert samples == plan_tile_samples(sizes, [0, 1], T, 0.1, full_frame=True)


def test_frame_sizes_may_be_a_mapping_of_pool_indices():
    by_index = {7: (544, 960), 9: (544, 960)}
    as_list = [(1, 1)] * 7 + [(544, 960), (1, 1), (544, 960)]
    assert plan_tile_samples(by_index, [7, 9], T, 0.1) == plan_tile_samples(
        as_list, [7, 9], T, 0.1
    )


def test_count_fills_small_pools_with_distinct_tiles():
    sizes = [(1080, 1920), (1080, 1920)]
    kw = {"extrema": {1: [(0, 0)]}, "count": 9}
    samples = plan_tile_samples(sizes, [0, 1], T, 0.1, **kw)
    assert len(samples) == 9
    per_frame = {
        i: [s.placement.index for s in samples if s.index == i] for i in (0, 1)
    }
    for tiles in per_frame.values():
        assert len(set(tiles)) == len(tiles)
        assert tiles == sorted(tiles)
    assert abs(len(per_frame[0]) - len(per_frame[1])) <= 1
    assert 0 in per_frame[1]
    assert [s.index for s in samples] == sorted(s.index for s in samples)
    assert samples == plan_tile_samples(sizes, [0, 1], T, 0.1, **kw)


def test_count_beyond_every_grid_tile_takes_them_all():
    n = len(tile_grid((544, 960), T, 0.1))
    samples = plan_tile_samples(
        [(544, 960)], [0], T, 0.1, count=500, full_frame=True
    )
    assert [s.placement.index for s in samples if s.placement] == list(
        range(n)
    )
    assert samples[-1].placement is None
    assert len(samples) == n + 1


def test_count_at_or_below_the_selection_keeps_one_tile_per_frame():
    sizes = [(544, 960), (544, 960)]
    one = plan_tile_samples(sizes, [0, 1], T, 0.1)
    assert plan_tile_samples(sizes, [0, 1], T, 0.1, count=2) == one
    assert plan_tile_samples(sizes, [0, 1], T, 0.1, count=1) == one
    assert len(one) == 2


def test_small_frames_are_letterboxed_on_their_tile():
    (sample,) = plan_tile_samples([(1080, 1920)], [0], (1088, 1088), 0.1)
    assert sample.placement.content is not None
