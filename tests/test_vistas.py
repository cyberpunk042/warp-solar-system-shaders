"""Smoke tests for the quality-round scenes.

Renders the three new ground-level vista scenes plus the four reworked hero
scenes at a tiny resolution and asserts the output is finite, in the display
range, and non-degenerate (real structure, not a flat fill). Keeps the round's
changes from silently regressing.

    python -m tests.test_vistas
"""

import numpy as np
import warp as wp

import warp_shaders as ws

# new this round + the four reworked scenes
_SCENES = ["ringed_vista", "binary_sea", "comet", "galaxy", "nebula",
           "mandelbulb", "aurora"]


def _check(name, time=0.0):
    img = ws.render(name, width=96, height=64, time=time)
    a = np.asarray(img, np.float32)
    assert a.shape == (64, 96, 3), (name, a.shape)
    assert np.all(np.isfinite(a)), f"{name}: non-finite pixels"
    assert a.min() >= 0.0, f"{name}: negative pixels"
    # non-degenerate: a real image has both dark and bright regions
    assert a.max() > 0.05, f"{name}: image is essentially black"
    assert a.std() > 0.01, f"{name}: image is a flat fill (no structure)"
    print(f"  {name}: OK  (min {a.min():.3f} max {a.max():.3f} std {a.std():.3f})")


def main():
    wp.init()
    # mandelbulb morphs power 2->8 over time; render it near the detailed end
    times = {"mandelbulb": 12.3, "aurora": 2.0, "binary_sea": 1.0}
    for name in _SCENES:
        _check(name, times.get(name, 0.0))
    print("ALL PASSED")


if __name__ == "__main__":
    main()
