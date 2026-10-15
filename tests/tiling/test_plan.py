from collections import Counter

import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    Mix,
    TilingConfig,
    batch_layout,
    gt_extent,
    plan_epoch,
)

N = 64


def _extents(m, empty_share, seed=0):
    rng = np.random.default_rng(seed)
    ext = rng.integers(1, 12, m)
    ext[rng.random(m) < empty_share] = 0
    return ext.tolist()


def test_gt_extent_counts_grid_tiles_touched_by_boxes():
    assert gt_extent((2268, 4032), [], (640, 640), 0.1) == 0
    one = gt_extent((2268, 4032), [[100, 100, 120, 120]], (640, 640), 0.1)
    spread = gt_extent(
        (2268, 4032),
        [[100, 100, 120, 120], [3900, 2100, 3920, 2120]],
        (640, 640),
        0.1,
    )
    assert one >= 1 and spread > one


def test_batch_layout():
    assert batch_layout(174, 64, 4) == [64] * 43 + [64]
    assert batch_layout(8, 64, 4) == [64, 64]
    assert batch_layout(1, 64, 4) == [32]
    assert batch_layout(5, 4, 8) == [
        4,
        2,
    ]  # F clamped to the batch; cap 2 for the last frame


@pytest.mark.parametrize(
    "frames,batch", [(6471, 64), (174, 64), (1000, 16), (63, 64), (1, 8)]
)
def test_frames_per_batch_at_batch_matches_untiled_steps(frames, batch):
    layout = batch_layout(frames, batch, batch)
    assert len(layout) == -(-frames // batch)
    assert all(n == batch for n in layout[:-1]) and layout[-1] <= batch


def test_every_frame_once_and_batches_fill():
    ext = _extents(174, 1 / 3)
    plan = plan_epoch(ext, N, TilingConfig(), epoch=0)
    frames = Counter()
    pos = 0
    for size in plan.batches:
        chunk = plan.tiles[pos : pos + size]
        pos += size
        per_frame = Counter(t.frame for t in chunk)
        assert all(1 <= c <= 2 * N // 4 for c in per_frame.values())
        frames.update(per_frame.keys())
    assert pos == len(plan.tiles) == sum(batch_layout(174, N, 4))
    assert sorted(frames) == list(range(174)) and set(frames.values()) == {1}


def test_frame_slots_are_contiguous_and_numbered():
    plan = plan_epoch(_extents(40, 0.3), N, TilingConfig(), epoch=1)
    seen = {}
    for i, t in enumerate(plan.tiles):
        if t.frame in seen:
            assert plan.tiles[i - 1].frame == t.frame
        seen.setdefault(t.frame, []).append(t.slot)
    assert all(slots == list(range(len(slots))) for slots in seen.values())
    assert all(
        plan.kinds(f) == tuple(t.kind for t in plan.tiles if t.frame == f)
        for f in seen
    )


def test_background_frames_get_only_background():
    ext = _extents(200, 0.4)
    plan = plan_epoch(ext, N, TilingConfig(), epoch=0)
    for t in plan.tiles:
        if ext[t.frame] == 0:
            assert t.kind == "background"


def test_realised_shares_track_mix_with_a_third_empty_frames():
    ext = _extents(3000, 1 / 3)
    kinds = Counter(
        t.kind
        for e in range(4)
        for t in plan_epoch(ext, N, TilingConfig(), epoch=e).tiles
    )
    total = sum(kinds.values())
    assert kinds["positive"] / total == pytest.approx(0.75, abs=0.02)
    assert kinds["background"] / total == pytest.approx(0.15, abs=0.02)
    assert kinds["full_frame"] / total == pytest.approx(0.10, abs=0.02)


def test_all_ground_truth_dataset_tracks_mix():
    ext = _extents(3000, 0.0)
    kinds = Counter(
        t.kind for t in plan_epoch(ext, N, TilingConfig(), epoch=0).tiles
    )
    assert kinds["background"] / sum(kinds.values()) == pytest.approx(
        0.15, abs=0.02
    )


def test_mostly_background_dataset():
    ext = _extents(400, 0.9)
    plan = plan_epoch(ext, N, TilingConfig(), epoch=0)
    assert sorted({t.frame for t in plan.tiles}) == list(range(400))
    assert all(size == N for size in plan.batches)
    assert plan.summary()["planned_background_share"] > 0.15


def test_all_background_dataset():
    plan = plan_epoch([0] * 50, N, TilingConfig(), epoch=0)
    assert {t.kind for t in plan.tiles} == {"background"}


def test_short_last_group():
    plan = plan_epoch([3, 0, 5, 2, 7], N, TilingConfig(), epoch=0)
    assert plan.batches == (64, 32)
    assert len(plan.tiles) == 96


def test_extent_weights_positive_slots():
    ext = [1, 20, 1, 20] * 200
    plan = plan_epoch(ext, N, TilingConfig(mix=Mix(1.0, 0.0, 0.0)), epoch=0)
    per_frame = Counter(t.frame for t in plan.tiles)
    small = np.mean([per_frame[f] for f in range(0, 800, 2)])
    large = np.mean([per_frame[f] for f in range(1, 800, 2)])
    assert large > 2 * small


def test_final_phase_has_no_full_frames():
    plan = plan_epoch(
        _extents(300, 0.2), N, TilingConfig(), epoch=0, final=True
    )
    assert all(t.kind != "full_frame" for t in plan.tiles)


def test_deterministic_per_epoch():
    ext = _extents(100, 0.3)
    a = plan_epoch(ext, N, TilingConfig(seed=3), epoch=2)
    assert a == plan_epoch(ext, N, TilingConfig(seed=3), epoch=2)
    assert a.tiles != plan_epoch(ext, N, TilingConfig(seed=3), epoch=3).tiles


def test_summary_keys():
    s = plan_epoch(_extents(100, 0.3), N, TilingConfig(), epoch=0).summary()
    assert set(s) == {
        "tiles_per_frame_mean",
        "tiles_per_frame_min",
        "tiles_per_frame_max",
        "background_frame_share",
        "planned_positive_share",
        "planned_background_share",
        "planned_full_frame_share",
    }
    assert s["tiles_per_frame_min"] >= 1


def test_groups_carry_the_dataset_background_ratio():
    ext = [0] * 100 + [3] * 200
    plan = plan_epoch(ext, N, TilingConfig(), epoch=0)
    order = list(dict.fromkeys(t.frame for t in plan.tiles))
    assert sorted(order) == list(range(300))
    for g in range(0, 300, 4):
        assert sum(ext[f] == 0 for f in order[g : g + 4]) in (1, 2)
