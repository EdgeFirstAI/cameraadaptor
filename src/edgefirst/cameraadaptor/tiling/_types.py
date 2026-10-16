# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Au-Zone Technologies
"""Type aliases shared by the tiling modules."""

from collections.abc import Sequence

from numpy.typing import ArrayLike

HW = Sequence[int]
"""A size as ``(H, W)`` in pixels."""

Rect = Sequence[int]
"""A rectangle as ``(x, y, w, h)`` in pixels."""

__all__ = ["HW", "ArrayLike", "Rect"]
