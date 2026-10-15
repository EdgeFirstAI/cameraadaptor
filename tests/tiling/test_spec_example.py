"""The `tiling` block example in TILING.md §6.1 passes the contract check."""

import pathlib

import pytest

from edgefirst.cameraadaptor.tiling import validate_tiling_block

yaml = pytest.importorskip("yaml")

SPEC = pathlib.Path(__file__).resolve().parents[2] / "TILING.md"


def test_metadata_example_validates():
    text = SPEC.read_text()
    start = text.index("```yaml\ntiling:") + len("```yaml\n")
    block = yaml.safe_load(text[start : text.index("```", start)])
    validate_tiling_block({**block, "input": {"shape": [1, 640, 640, 3]}})
