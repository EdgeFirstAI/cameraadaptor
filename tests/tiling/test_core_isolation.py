"""The tiling core needs only NumPy; OpenCV and PyTorch are extras."""

import ast
import pathlib
import subprocess
import sys

import pytest

import edgefirst.cameraadaptor.tiling as core

CORE = pathlib.Path(core.__file__).parent
TORCH_OPS = CORE.parent / "pytorch" / "tiling.py"
STDLIB = {
    "__future__",
    "copy",
    "dataclasses",
    "functools",
    "math",
    "struct",
    "typing",
    "collections",
    "numbers",
    "re",
}


def _files():
    return [*sorted(CORE.glob("*.py")), TORCH_OPS]


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 1 and path.parent == CORE:
                continue
            assert node.level == 0, f"{path.name}: relative import escapes"
            yield node.module


def _allowed(path):
    allowed = STDLIB | {"numpy"}
    if path.name == "pixels.py":
        allowed |= {"cv2"}
    if path == TORCH_OPS:
        allowed |= {"torch", "edgefirst"}
    return allowed


def test_tiling_modules_import_only_their_dependencies():
    for path in _files():
        for name in _imports(path):
            root = name.split(".")[0]
            assert root in _allowed(path), f"{path.name} imports {name}"
            if root == "edgefirst":
                assert name.startswith("edgefirst.cameraadaptor.tiling"), name


@pytest.mark.parametrize(
    "module",
    ["edgefirst.cameraadaptor.tiling", "edgefirst.cameraadaptor.calibration"],
)
def test_importing_the_core_loads_neither_opencv_nor_torch(module):
    code = (
        f"import sys, {module}; "
        "print(sorted({'cv2', 'torch', 'tensorflow'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "[]"
