"""Smoke tests for the GPU-singularity round.

Renders each scene at a moment of its arc (they animate over ``--frames``) at a
tiny resolution and asserts finite, in-range, non-degenerate output — the whole
detonation arc, the single-block roof-pierce, the PCIe ignition, and the void
aftermath.

    python -m tests.test_singularity
"""

import numpy as np
import warp as wp

import warp_shaders as ws

# scene -> a time inside its arc where there is clearly something on screen
_SCENES = [
    ("gpu_singularity", 7.5),   # the mushroom cloud off the real board
    ("memory_overflow", 5.0),   # block detonation + rising mushroom
    ("power_draw", 3.6),        # ignition — electrons through the real board
    ("mind_escape", 1.5),       # the mind loose in the void
]


def _check(name, time):
    img = np.asarray(ws.render(name, width=96, height=72, time=time), np.float32)
    assert img.shape == (72, 96, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.05, f"{name}: essentially black"
    assert img.std() > 0.01, f"{name}: flat fill"
    print(f"  {name}: OK  (max {img.max():.3f} std {img.std():.3f})")


def main():
    wp.init()
    for n, t in _SCENES:
        _check(n, t)
    print(f"ALL PASSED ({len(_SCENES)} scenes)")


if __name__ == "__main__":
    main()
