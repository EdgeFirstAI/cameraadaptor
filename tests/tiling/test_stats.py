import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    Mix,
    SamplerStats,
    TilingConfig,
    sample_windows,
)


def test_summary_reports_shares_and_rates():
    stats = SamplerStats()
    windows = sample_windows(
        (2160, 3840),
        [[1000, 500, 1030, 530], [1100, 500, 1130, 530]],
        [0, 0],
        TilingConfig(grid_prob=0.0),
        np.random.default_rng(0),
        k=4000,
    )
    for w in windows:
        stats.record(w)
    stats.add("decodes", 1000)
    stats.add("samples", 1000)
    s = stats.summary()
    assert s["positive_share"] + s["background_share"] + s[
        "full_frame_share"
    ] == pytest.approx(1.0)
    assert 0.0 <= s["empty_rate"] <= 1.0
    assert s["decodes_per_sample"] == 1.0
    assert s["clipped_share"] >= 0.0
    assert s["native_share"] == pytest.approx(0.5, abs=0.03)


def test_vector_round_trip_and_add():
    a = SamplerStats()
    a.add("windows", 3)
    b = SamplerStats(a.v)
    b += a
    assert b.v[SamplerStats.index("windows")] == 6
    assert len(b.v) == SamplerStats.SIZE


def test_empty_summary_is_zero_not_nan():
    assert all(v == 0.0 for v in SamplerStats().summary().values())


def test_anchor_histogram_counts_positive_windows():
    cfg = TilingConfig(mix=Mix(1.0, 0.0, 0.0))
    stats = SamplerStats()
    for w in sample_windows(
        (2160, 3840),
        [[1000, 500, 1030, 530]],
        [0],
        cfg,
        np.random.default_rng(1),
        k=100,
    ):
        stats.record(w)
    assert stats.anchor_histogram().sum() == 100


def test_objects_histogram_sum_equals_windows():
    cfg = TilingConfig(mix=Mix(1.0, 0.0, 0.0))
    stats = SamplerStats()
    windows = sample_windows(
        (2160, 3840),
        [[1000, 500, 1030, 530]],
        [0],
        cfg,
        np.random.default_rng(1),
        k=100,
    )
    for w in windows:
        stats.record(w)
    assert stats.objects_histogram().sum() == 100


def test_scale_histogram_sum_equals_non_native_non_full_frame():
    cfg = TilingConfig(mix=Mix(1.0, 0.0, 0.0))
    stats = SamplerStats()
    windows = sample_windows(
        (2160, 3840),
        [[1000, 500, 1030, 530]],
        [0],
        cfg,
        np.random.default_rng(2),
        k=100,
    )
    for w in windows:
        stats.record(w)
    non_native = (
        stats.v[SamplerStats.index("windows")]
        - stats.v[SamplerStats.index("full_frame")]
        - stats.v[SamplerStats.index("native")]
    )
    assert stats.scale_histogram().sum() == pytest.approx(non_native)


def test_summary_has_dropped_per_tile_keys():
    stats = SamplerStats()
    summary = stats.summary()
    assert "dropped_visibility_per_tile" in summary
    assert "dropped_min_px_per_tile" in summary
    assert "dropped_cap_per_tile" in summary
    assert summary["dropped_visibility_per_tile"] == 0.0
    assert summary["dropped_min_px_per_tile"] == 0.0
    assert summary["dropped_cap_per_tile"] == 0.0


def test_objects_per_positive_tile():
    cfg = TilingConfig(mix=Mix(1.0, 0.0, 0.0))
    boxes = [[1000, 500, 1030, 530], [1040, 500, 1070, 530]]
    windows = sample_windows(
        (2160, 3840), boxes, [0, 0], cfg, np.random.default_rng(3), k=50
    )
    stats = SamplerStats()
    for w in windows:
        stats.record(w)
    expected = np.mean([len(w.boxes) for w in windows if w.kind == "positive"])
    assert stats.summary()["objects_per_positive_tile"] == pytest.approx(
        expected
    )
