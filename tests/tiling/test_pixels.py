import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from edgefirst.cameraadaptor.tiling.pixels import render_window  # noqa: E402


def _frame(h=720, w=1280):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_native_window_is_an_exact_crop():
    f = _frame()
    out = render_window(f, (100, 50, 320, 320), (320, 320))
    np.testing.assert_array_equal(out, f[50:370, 100:420])
    assert out.flags["C_CONTIGUOUS"]


def test_downscale_is_area_filtered():
    f = _frame()
    out = render_window(f, (0, 0, 640, 640), (320, 320))
    np.testing.assert_array_equal(
        out,
        cv2.resize(f[:640, :640], (320, 320), interpolation=cv2.INTER_AREA),
    )


def test_bilinear_mode_and_upscale():
    f = _frame()
    down = render_window(f, (0, 0, 640, 640), (320, 320), resize="bilinear")
    np.testing.assert_array_equal(
        down,
        cv2.resize(f[:640, :640], (320, 320), interpolation=cv2.INTER_LINEAR),
    )
    up = render_window(f, (0, 0, 160, 160), (320, 320))
    np.testing.assert_array_equal(
        up,
        cv2.resize(f[:160, :160], (320, 320), interpolation=cv2.INTER_LINEAR),
    )


def test_letterbox_pads_with_114_centred():
    f = _frame(300, 400)
    out = render_window(
        f, (0, 0, 400, 300), (640, 640), content=(0, 80, 640, 480)
    )
    assert out.shape == (640, 640, 3)
    assert (out[:80] == 114).all() and (out[560:] == 114).all()
