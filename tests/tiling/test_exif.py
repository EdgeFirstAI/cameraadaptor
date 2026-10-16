import struct

import numpy as np
import pytest

from edgefirst.cameraadaptor.tiling.exif import orientation, oriented_hw

cv2 = pytest.importorskip("cv2")

HW = (24, 40)


def _exif_segment(value, byte_order):
    """APP1 segment holding one IFD0 entry: Orientation (SHORT) = value."""
    e = byte_order
    mark = b"II" if e == "<" else b"MM"
    tiff = mark + struct.pack(e + "HI", 42, 8)
    tiff += struct.pack(e + "H", 1)
    tiff += struct.pack(e + "HHIHH", 0x0112, 3, 1, value, 0)
    tiff += struct.pack(e + "I", 0)
    payload = b"Exif\x00\x00" + tiff
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def _jpeg(value=None, byte_order="<"):
    ok, buf = cv2.imencode(".jpg", np.zeros((*HW, 3), np.uint8))
    assert ok
    data = buf.tobytes()
    if value is None:
        return data
    return data[:2] + _exif_segment(value, byte_order) + data[2:]


@pytest.mark.parametrize("value", range(1, 9))
@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_orientation_is_read_from_jpeg_exif(value, byte_order):
    assert orientation(_jpeg(value, byte_order)) == value


def test_frames_without_exif_or_jpeg_are_upright():
    assert orientation(_jpeg()) == 1
    ok, png = cv2.imencode(".png", np.zeros((4, 4, 3), np.uint8))
    assert ok
    assert orientation(png.tobytes()) == 1
    assert orientation(b"") == 1


@pytest.mark.parametrize("value", range(1, 9))
def test_oriented_size_matches_opencv_decode(value, tmp_path):
    path = tmp_path / "f.jpg"
    path.write_bytes(_jpeg(value))
    decoded = cv2.imread(str(path)).shape[:2]
    assert oriented_hw(HW, orientation(path.read_bytes())) == decoded
