"""Tests for compatibility utilities."""

import pytest

from edgefirst.cameraadaptor._compat import (
    PY38,
    PY39,
    PY310,
    check_lightning_available,
    check_opencv_available,
    check_tensorflow_available,
    check_torch_available,
    require_opencv,
    require_tensorflow,
    require_torch,
)


class TestPythonVersionChecks:
    """Tests for Python version constants."""

    def test_py38_is_bool(self):
        """Test PY38 is a boolean."""
        assert isinstance(PY38, bool)

    def test_py39_is_bool(self):
        """Test PY39 is a boolean."""
        assert isinstance(PY39, bool)

    def test_py310_is_bool(self):
        """Test PY310 is a boolean."""
        assert isinstance(PY310, bool)

    def test_version_ordering(self):
        """Test that version checks are ordered correctly."""
        # If PY310 is True, PY39 and PY38 must also be True
        if PY310:
            assert PY39
            assert PY38
        # If PY39 is True, PY38 must also be True
        if PY39:
            assert PY38


class TestCheckFunctions:
    """Tests for availability check functions."""

    def test_check_torch_available_returns_bool(self):
        """Test check_torch_available returns a boolean."""
        result = check_torch_available()
        assert isinstance(result, bool)

    def test_check_tensorflow_available_returns_bool(self):
        """Test check_tensorflow_available returns a boolean."""
        result = check_tensorflow_available()
        assert isinstance(result, bool)

    def test_check_lightning_available_returns_bool(self):
        """Test check_lightning_available returns a boolean."""
        result = check_lightning_available()
        assert isinstance(result, bool)

    def test_check_opencv_available_returns_bool(self):
        """Test check_opencv_available returns a boolean."""
        result = check_opencv_available()
        assert isinstance(result, bool)


class TestRequireFunctions:
    """Tests for require functions."""

    def test_require_torch_with_torch_installed(self):
        """Test require_torch passes when torch is installed."""
        pytest.importorskip("torch")
        require_torch()  # Should not raise

    def test_require_tensorflow_with_tensorflow_installed(self):
        """Test require_tensorflow passes when tensorflow is installed."""
        pytest.importorskip("tensorflow")
        require_tensorflow()  # Should not raise

    def test_require_opencv_with_opencv_installed(self):
        """Test require_opencv passes when opencv is installed."""
        pytest.importorskip("cv2")
        require_opencv()  # Should not raise


class TestCheckFunctionsReturnsTrue:
    """Tests that check functions return True when libraries are installed."""

    def test_check_torch_returns_true(self):
        """Test check_torch_available returns True when torch is installed."""
        pytest.importorskip("torch")
        assert check_torch_available() is True

    def test_check_tensorflow_returns_true(self):
        """Test check_tensorflow_available returns True when installed."""
        pytest.importorskip("tensorflow")
        assert check_tensorflow_available() is True

    def test_check_opencv_returns_true(self):
        """Test check_opencv_available returns True when cv2 is installed."""
        pytest.importorskip("cv2")
        assert check_opencv_available() is True
