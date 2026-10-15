import pytest

from edgefirst.cameraadaptor.tiling import SPEC_VERSION, Mix, TilingConfig


def test_defaults_match_spec_table():
    cfg = TilingConfig()
    assert cfg.frames_per_batch == 4
    assert (cfg.mix.positive, cfg.mix.background, cfg.mix.full_frame) == (
        0.75,
        0.15,
        0.10,
    )
    assert cfg.grid_prob == 0.25 and cfg.cut_prob == 0.1
    assert cfg.scale.native_prob == 0.5 and cfg.scale.range == (0.75, 1.333)
    assert cfg.targets.min_visibility == 0.25 and cfg.targets.min_box_px == 2.0
    assert cfg.targets.truncation == "drop"
    assert cfg.final_phase_epochs == 10 and cfg.grid_min_overlap == 0.1


def test_windows_per_frame_is_gone():
    with pytest.raises(ValueError, match="windows_per_frame"):
        TilingConfig.from_dict({"windows_per_frame": 4})


def test_from_dict_accepts_nested_and_dotted_keys():
    cfg = TilingConfig.from_dict(
        {
            "tile_size": 320,
            "mix.positive": 0.6,
            "scale": {"native_prob": 1.0, "range": [0.8, 1.25]},
        }
    )
    assert cfg.tile_size == (320, 320)
    assert cfg.mix.positive == 0.6 and cfg.mix.background == 0.15
    assert cfg.scale.native_prob == 1.0 and cfg.scale.range == (0.8, 1.25)


def test_round_trip():
    cfg = TilingConfig.from_dict(
        {"tile_size": [512, 512], "mix.full_frame": 0.0, "seed": 7}
    )
    assert TilingConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "bad",
    [
        {"frames_per_batch": 0},
        {"mix.positive": -0.1},
        {"grid_prob": 1.5},
        {"scale.range": [1.1, 1.3]},
        {"targets.truncation": "clip"},
        {"grid_min_overlap": 1.0},
        {"mix": {"positive": 0, "background": 0, "full_frame": 0}},
    ],
)
def test_invalid_values_raise(bad):
    with pytest.raises(ValueError):
        TilingConfig.from_dict(bad)


def test_unknown_keys_raise():
    with pytest.raises(ValueError, match="unknown tiling parameters"):
        TilingConfig.from_dict({"object_fraction": 0.5})
    with pytest.raises(ValueError, match="mix.negative"):
        TilingConfig.from_dict({"mix.negative": 0.1})


def test_mix_shares_redistribute():
    assert Mix().shares() == pytest.approx((0.75, 0.15, 0.10))
    assert Mix().shares(has_objects=False) == pytest.approx((0.0, 0.90, 0.10))
    assert Mix().shares(final=True) == pytest.approx((0.85, 0.15, 0.0))
    assert Mix(0.5, 0.0, 0.0).shares(has_objects=False) == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_batch_sampling_defaults():
    cfg = TilingConfig()
    assert cfg.frames_per_batch == 4 and cfg.placement_candidates == 4
    assert SPEC_VERSION == "3.1"


@pytest.mark.parametrize(
    "bad", [{"frames_per_batch": 0}, {"placement_candidates": 0}]
)
def test_batch_sampling_fields_validate(bad):
    with pytest.raises(ValueError):
        TilingConfig.from_dict(bad)


def test_batch_sampling_fields_round_trip():
    cfg = TilingConfig.from_dict(
        {"frames_per_batch": 8, "placement_candidates": 2}
    )
    assert TilingConfig.from_dict(cfg.to_dict()) == cfg
