# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""EXIF orientation of encoded frames (TILING.md §4.3).

Frames are used as decoded with their EXIF orientation applied, which is what
OpenCV's ``imread`` (and so the Ultralytics trainer) does. Box labels refer to
the oriented frame. A decoder that ignores EXIF, such as TensorFlow's, must
apply ``orientation`` itself; ``oriented_hw`` gives the decoded size without
decoding, for planning and calibration.
"""

from __future__ import annotations

import struct

_ORIENTATION_TAG = 0x0112
_TRANSPOSED = (5, 6, 7, 8)


def orientation(data: bytes) -> int:
    """EXIF orientation (1–8) of a JPEG, or 1 when it has none or the data
    is not a JPEG.
    """
    if data[:2] != b"\xff\xd8":
        return 1
    pos = 2
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker in (0xD9, 0xDA):  # end of image, start of scan
            return 1
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        segment = data[pos + 4 : pos + 2 + length]
        if marker == 0xE1 and segment[:6] == b"Exif\x00\x00":
            return _tiff_orientation(segment[6:])
        pos += 2 + length
    return 1


def _tiff_orientation(tiff: bytes) -> int:
    if len(tiff) < 8 or tiff[:2] not in (b"II", b"MM"):
        return 1
    endian = "<" if tiff[:2] == b"II" else ">"
    (ifd,) = struct.unpack(endian + "I", tiff[4:8])
    if ifd + 2 > len(tiff):
        return 1
    (count,) = struct.unpack(endian + "H", tiff[ifd : ifd + 2])
    for i in range(count):
        entry = tiff[ifd + 2 + 12 * i : ifd + 14 + 12 * i]
        if len(entry) < 12:
            break
        tag, kind = struct.unpack(endian + "HH", entry[:4])
        if tag == _ORIENTATION_TAG and kind == 3:  # SHORT
            (value,) = struct.unpack(endian + "H", entry[8:10])
            return value if 1 <= value <= 8 else 1
    return 1


def oriented_hw(stored_hw: tuple[int, int], value: int) -> tuple[int, int]:
    """``(H, W)`` after applying EXIF orientation ``value`` to a frame stored
    as ``stored_hw``.
    """
    h, w = stored_hw
    return (w, h) if value in _TRANSPOSED else (h, w)


__all__ = ["orientation", "oriented_hw"]
