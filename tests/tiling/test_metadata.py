import copy

import pytest

from edgefirst.cameraadaptor.tiling import (
    SPEC_VERSION,
    TilingConfig,
    build_tiling_block,
    reexport_tiling_block,
    validate_tiling_block,
)
from edgefirst.cameraadaptor.tiling.contract import ContractError

CFG = TilingConfig(enabled=True, tile_size=(640, 640), seed=3)


def _doc(block, hw):
    return {"input": {"shape": [1, 3, *hw]}, "tiling": block}


def test_fresh_block_is_a_valid_tiled_section():
    block = build_tiling_block(CFG, session="t-1a2b")
    validate_tiling_block(_doc(block, (640, 640)))
    assert block["mode"] == "tiled" and block["tile"] == [640, 640]
    assert block["training"] == {
        "session": "t-1a2b",
        "sampler": "edgefirst-tiling",
        "version": SPEC_VERSION,
        "params": CFG.to_dict(),
    }
    assert block["export"] == {
        "session": "t-1a2b",
        "calibration": {
            "resize": "tiled",
            "input_shape": [640, 640],
            "full_frame": False,
        },
    }
    assert "enabled" not in block and "deployment_hint" not in block
    assert block["grid"] == {
        "algorithm": "evendist",
        "version": 1,
        "min_overlap": 0.1,
    }


def test_end2end_heads_skip_per_tile_nms():
    assert (
        build_tiling_block(CFG, session="t-1", per_tile_nms="none")[
            "per_tile"
        ]["nms"]
        == "none"
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"per_tile_nms": "soft"},
        {"merge_mode": "average"},
        {"merge_metric": "giou"},
    ],
)
def test_rejects_unknown_enumerations(kw):
    with pytest.raises(ValueError):
        build_tiling_block(CFG, session="t-1", **kw)


def test_full_frame_flag_reaches_the_calibration_record():
    block = build_tiling_block(CFG, session="t-1", full_frame=True)
    assert (
        block["full_frame"]["enabled"] is True
        and block["export"]["calibration"]["full_frame"] is True
    )
    validate_tiling_block(_doc(block, (640, 640)))


def test_whole_frame_reexport_keeps_training_and_tile():
    source = build_tiling_block(CFG, session="t-1a2b", full_frame=True)
    before = copy.deepcopy(source)
    block = reexport_tiling_block(
        source,
        mode="whole_frame",
        input_hw=(2176, 3840),
        export_session="t-3c4d",
        weights_from="t-1a2b",
    )
    validate_tiling_block(_doc(block, (2176, 3840)))
    assert source == before
    assert block["mode"] == "whole_frame" and block["tile"] == [640, 640]
    assert block["training"] == source["training"]
    assert block["full_frame"]["enabled"] is False
    assert block["export"] == {
        "session": "t-3c4d",
        "weights_from": "t-1a2b",
        "calibration": {
            "resize": "letterbox",
            "input_shape": [2176, 3840],
            "full_frame": False,
        },
    }


def test_tiled_reexport_takes_the_new_tile():
    source = build_tiling_block(CFG, session="t-1a2b", full_frame=True)
    block = reexport_tiling_block(
        source,
        mode="tiled",
        input_hw=(1280, 1280),
        export_session="t-3c4e",
        weights_from="t-1a2b",
    )
    validate_tiling_block(_doc(block, (1280, 1280)))
    assert (
        block["tile"] == [1280, 1280]
        and block["full_frame"]["enabled"] is True
    )
    assert block["export"]["calibration"] == {
        "resize": "tiled",
        "input_shape": [1280, 1280],
        "full_frame": True,
    }


def test_reexport_keeps_unknown_keys_and_rejects_bad_mode():
    source = dict(build_tiling_block(CFG, session="t-1"), extra={"k": 1})
    assert reexport_tiling_block(
        source,
        mode="tiled",
        input_hw=(640, 640),
        export_session="t-2",
        weights_from="t-1",
    )["extra"] == {"k": 1}
    with pytest.raises(ValueError):
        reexport_tiling_block(
            source,
            mode="auto",
            input_hw=(640, 640),
            export_session="t-2",
            weights_from="t-1",
        )


def test_invalid_session_fails_validation():
    with pytest.raises(ContractError):
        validate_tiling_block(
            _doc(build_tiling_block(CFG, session=""), (640, 640))
        )
