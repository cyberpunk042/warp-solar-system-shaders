"""Smoke test for the engine-leap (path-traced) scenes.

Renders each at a tiny resolution (few samples per pixel) and asserts finite,
in-range, non-degenerate output.

    python -m tests.test_engine_leap
"""

import numpy as np
import warp as wp

import warp_shaders as ws

_SCENES = ["cornell_box", "glass_box"]


def _check(name):
    img = np.asarray(ws.render(name, width=96, height=72, time=0.0), np.float32)
    assert img.shape == (72, 96, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.05, f"{name}: essentially black"
    assert img.std() > 0.01, f"{name}: flat fill"
    print(f"  {name}: OK  (max {img.max():.3f} std {img.std():.3f})")


def main():
    wp.init()
    for n in _SCENES:
        _check(n)
    print(f"ALL PASSED ({len(_SCENES)} scenes)")


if __name__ == "__main__":
    main()
