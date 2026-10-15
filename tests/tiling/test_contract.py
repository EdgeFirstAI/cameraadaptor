"""Tiling deployment contract conformance (usermanual metadata.md#tiling,
calibration.md).
"""

import copy
import json
import pathlib

import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling import (
    ContractError,
    UnsupportedVersion,
    check_converter_tiling,
    check_snapshot_geometry,
    check_snapshot_params,
    model_input_hw,
    parse_session_id,
    validate_tiling_block,
)

FIXTURE = json.loads(
    (
        pathlib.Path(__file__).parents[1] / "fixtures" / "tiling_contract.json"
    ).read_text()
)
VALID = {case["name"]: case["metadata"] for case in FIXTURE["metadata_valid"]}


def merge_patch(target, patch):
    """RFC 7386 JSON merge patch."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


@pytest.mark.parametrize(
    "case", FIXTURE["metadata_valid"], ids=lambda c: c["name"]
)
def test_valid_metadata_passes(case):
    validate_tiling_block(case["metadata"])


@pytest.mark.parametrize(
    "case", FIXTURE["metadata_invalid"], ids=lambda c: c["name"]
)
def test_invalid_metadata_is_rejected(case):
    metadata = merge_patch(VALID[case["base"]], case["patch"])
    with pytest.raises(ContractError, match=case["error"]):
        validate_tiling_block(metadata)


@pytest.mark.parametrize(
    "case", FIXTURE["snapshot_geometry"], ids=lambda c: c["name"]
)
def test_snapshot_geometry(case):
    args = (
        case["params"],
        case["sample_hw"],
        case["model_input_hw"],
        case["tiling_mode"],
        case.get("full_frame", False),
    )
    if case["error"] is None:
        check_snapshot_geometry(*args)
    else:
        with pytest.raises(ContractError, match=case["error"]):
            check_snapshot_geometry(*args)


@pytest.mark.parametrize("value,expected", FIXTURE["session_ids"]["valid"])
def test_session_ids(value, expected):
    assert parse_session_id(value) == expected


@pytest.mark.parametrize("value", FIXTURE["session_ids"]["invalid"])
def test_invalid_session_ids(value):
    with pytest.raises(ContractError):
        parse_session_id(value)


@pytest.mark.parametrize(
    "shape,hw",
    [
        ([1, 3, 480, 640], (480, 640)),
        ([1, 480, 640, 3], (480, 640)),
        ([1, 1, 64, 32], (64, 32)),
        ([1, 64, 32, 4], (64, 32)),
    ],
)
def test_model_input_hw_reads_both_layouts(shape, hw):
    assert model_input_hw(shape) == hw


def test_model_input_hw_rejects_ambiguous_or_bad_shapes():
    for shape in ([1, 640, 640], [1, 64, 64, 64]):
        with pytest.raises(ContractError):
            model_input_hw(shape)


@pytest.mark.parametrize(
    "shape",
    [
        None,
        640,
        "1x3x640x640",
        {"n": 1},
        [1, 3, None, 640],
        [1, 3, 640.0, 640],
        [1, 3, "640", 640],
        [1, 3, True, 640],
    ],
)
def test_model_input_hw_raises_contract_error_on_malformed_shapes(shape):
    with pytest.raises(ContractError):
        model_input_hw(shape)


def test_model_input_hw_accepts_numpy():
    assert model_input_hw(np.array([1, 3, 480, 640])) == (480, 640)
    assert model_input_hw(
        (np.int64(1), np.int64(480), np.int64(640), np.int64(3))
    ) == (480, 640)


def test_snapshot_geometry_accepts_numpy_sizes():
    params = {
        "input_shape": [np.int64(640), np.int64(640)],
        "resize": "letterbox",
    }
    tensor_shape = np.zeros((2, 3, 640, 640), dtype=np.uint8).shape
    check_snapshot_geometry(
        params, tensor_shape[2:4], np.array([640, 640]), None
    )
    check_snapshot_geometry(
        params,
        np.array(tensor_shape)[2:4],
        (np.int64(640), np.int64(640)),
        None,
    )


@pytest.mark.parametrize(
    "bad", [None, [640.0, 640], ["640", 640], "640", [640], [640, 640, 3]]
)
@pytest.mark.parametrize(
    "slot", ["input_shape", "sample_hw", "model_input_hw"]
)
def test_snapshot_geometry_rejects_malformed_sizes(bad, slot):
    args = {
        "params": {"input_shape": [640, 640], "resize": "letterbox"},
        "sample_hw": [640, 640],
        "model_input_hw": [640, 640],
    }
    if slot == "input_shape":
        args["params"] = {"input_shape": bad, "resize": "letterbox"}
    else:
        args[slot] = bad
    with pytest.raises(ContractError):
        check_snapshot_geometry(
            args["params"], args["sample_hw"], args["model_input_hw"], None
        )


def test_unsupported_version_is_a_distinct_contract_error():
    metadata = merge_patch(
        VALID["tiled_as_trained"], {"tiling": {"version": 2}}
    )
    with pytest.raises(UnsupportedVersion):
        validate_tiling_block(metadata)
    assert issubclass(UnsupportedVersion, ContractError)
    for bad in ("1", 1.0, None):
        with pytest.raises(ContractError) as info:
            validate_tiling_block(
                merge_patch(
                    VALID["tiled_as_trained"], {"tiling": {"version": bad}}
                )
            )
        assert not isinstance(info.value, UnsupportedVersion)


def _required_paths(block, prefix=()):
    for key, value in block.items():
        path = prefix + (key,)
        if path in {("export", "weights_from"), ("training", "params")}:
            continue
        yield path
        if isinstance(value, dict) and path != ("training", "params"):
            yield from _required_paths(value, path)


def _without(metadata, path):
    result = copy.deepcopy(metadata)
    node = result["tiling"]
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return result


@pytest.mark.parametrize(
    "path",
    sorted(set(_required_paths(VALID["whole_frame_reexport"]["tiling"]))),
    ids=".".join,
)
def test_every_v1_key_is_required(path):
    with pytest.raises(ContractError):
        validate_tiling_block(_without(VALID["whole_frame_reexport"], path))


def test_optional_keys_may_be_omitted():
    metadata = _without(
        VALID["whole_frame_reexport"], ("export", "weights_from")
    )
    metadata["tiling"]["training"]["params"] = {}
    validate_tiling_block(metadata)


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"grid": {"min_overlap": 1}}, "tiling.grid.min_overlap"),
        ({"grid": {"min_overlap": -0.1}}, "tiling.grid.min_overlap"),
        ({"per_tile": {"iou": 1.5}}, "tiling.per_tile.iou"),
        ({"per_tile": {"score": -1}}, "tiling.per_tile.score"),
        ({"merge": {"threshold": 2}}, "tiling.merge.threshold"),
        ({"merge": {"class_agnostic": 0}}, "tiling.merge.class_agnostic"),
        ({"merge": {"max_det": 300.0}}, "tiling.merge.max_det"),
        ({"merge": {"max_det": True}}, "tiling.merge.max_det"),
        ({"pad": [114, 114]}, "tiling.pad"),
        ({"pad": [114, 114, 256]}, "tiling.pad"),
        ({"pad": [114, 114, 114.0]}, "tiling.pad"),
        ({"full_frame": {"enabled": "no"}}, "tiling.full_frame.enabled"),
        (
            {"full_frame": {"own_above_area": 1.2}},
            "tiling.full_frame.own_above_area",
        ),
        (
            {"export": {"calibration": {"full_frame": 0}}},
            "tiling.export.calibration.full_frame",
        ),
        ({"training": {"session": "garbage"}}, "tiling.training.session"),
        ({"training": {"sampler": 3}}, "tiling.training.sampler"),
        ({"training": {"params": "x"}}, "tiling.training.params"),
        ({"export": {"weights_from": "t-"}}, "tiling.export.weights_from"),
    ],
)
def test_field_ranges(patch, field):
    metadata = merge_patch(VALID["whole_frame_reexport"], {"tiling": patch})
    with pytest.raises(ContractError, match=field.replace(".", r"\.")):
        validate_tiling_block(metadata)


def test_range_boundaries_are_accepted():
    metadata = merge_patch(
        VALID["tiled_as_trained"],
        {
            "tiling": {
                "grid": {"min_overlap": 0},
                "pad": [0, 128, 255],
                "per_tile": {"iou": 1, "score": 0},
                "merge": {"threshold": 0, "max_det": 1},
                "full_frame": {"enabled": True, "own_above_area": 1},
                "export": {"calibration": {"full_frame": True}},
            }
        },
    )
    validate_tiling_block(metadata)
    metadata["tiling"]["tile"] = np.array([640, 640])
    metadata["tiling"]["pad"] = np.array([114, 114, 114])
    metadata["input"]["shape"] = np.array([1, 3, 640, 640])
    validate_tiling_block(metadata)


@pytest.mark.parametrize(
    "case", FIXTURE["converter_tiling"], ids=lambda c: c["name"]
)
def test_converter_tiling(case):
    document = {"tiling": case["tiling"]}
    if case["error"] is None:
        check_converter_tiling(document)
    else:
        with pytest.raises(ContractError, match=case["error"]):
            check_converter_tiling(document)


def test_converter_tiling_absent_key_passes():
    check_converter_tiling({})
    check_converter_tiling({"input": {"shape": [1, 3, 640, 640]}})


@pytest.mark.parametrize("params", [None, [], "params", 3])
def test_snapshot_params_rejects_non_mapping(params):
    with pytest.raises(ContractError, match="params"):
        check_snapshot_params(params, [640, 640])


@pytest.mark.parametrize(
    "params,match",
    [
        ({"resize": "letterbox"}, "input_shape"),
        ({"input_shape": [640, 640]}, "resize"),
    ],
)
def test_snapshot_params_rejects_missing_keys(params, match):
    with pytest.raises(ContractError, match=match):
        check_snapshot_params(params, [640, 640])


def test_snapshot_params_rejects_mismatched_input_shape():
    with pytest.raises(ContractError, match="does not match the model input"):
        check_snapshot_params(
            {"input_shape": [640, 640], "resize": "letterbox"}, [2176, 3840]
        )


def test_snapshot_params_returns_snapshot_hw():
    assert check_snapshot_params(
        {"input_shape": [640, 480], "resize": "letterbox"},
        np.array([640, 480]),
    ) == (640, 480)


def test_snapshot_geometry_checks_params_before_sample():
    with pytest.raises(ContractError, match="does not match the model input"):
        check_snapshot_geometry(
            {"input_shape": [640, 640], "resize": "letterbox"},
            "not-a-size",
            [2176, 3840],
            None,
        )


@pytest.mark.parametrize("version", [1.0, True, "1"])
def test_non_integer_version_is_rejected_by_producer_check(version):
    metadata = merge_patch(
        VALID["tiled_as_trained"], {"tiling": {"version": version}}
    )
    with pytest.raises(ContractError, match="tiling.version"):
        validate_tiling_block(metadata)
