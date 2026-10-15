"""Append random EvenDist cases computed by HAL to
tests/fixtures/tiling_golden.json.

Run from a venv with HAL's ``edgefirst-image`` bindings installed (the
``dev`` extra pins them):
    python tests/tiling/gen_golden_from_hal.py

The HAL version that produced the cases is recorded beside them.
"""

import json
import pathlib
import random

from edgefirst.image import (  # tile_grid(frame_w, frame_h, tile_w, tile_h, r)
    tile_grid,
    version,
)

FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "tiling_golden.json"


def main(n: int = 200, seed: int = 20260928) -> None:
    rng = random.Random(seed)
    golden = json.loads(FIXTURE.read_text())
    cases = []
    for _ in range(n):
        t = rng.choice([256, 320, 512, 640, 1024])
        length = rng.randint(t // 2, 8 * t)
        r = round(rng.choice([0.0, 0.1, 0.15, 0.2, 0.25, 0.333, 0.5, 0.9]), 3)
        # frame_h = tile height (single row) so unique x origins are the axis
        # origins.
        tiles = tile_grid(length, t, t, t, r)
        origins = sorted({spec.source.x for spec in tiles})
        cases.append({"L": length, "t": t, "r": r, "origins": origins})
    golden["hal_random"] = cases
    golden["hal_random_version"] = version()
    FIXTURE.write_text(json.dumps(golden, indent=2) + "\n")


if __name__ == "__main__":
    main()
