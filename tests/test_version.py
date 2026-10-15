"""The runtime version matches the one declared in pyproject.toml."""

import pathlib
import re

import edgefirst.cameraadaptor as ca

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject():
    declared = re.search(
        r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.MULTILINE
    )
    assert declared is not None
    assert ca.__version__ == declared.group(1)
