"""Smoke tests for the electricity-in-motion scenes.

Renders each at a moment of its arc at a tiny resolution and asserts finite,
in-range, non-degenerate output — the lightning bolt, the Tesla coil, the spark
gap, the plasma globe, the charging capacitor, the spinning motor, the transformer,
and the power grid.

    python -m tests.test_electric
"""

import numpy as np
import warp as wp

import warp_shaders as ws

_SCENES = [
    ("lightning", 0.12),
    ("tesla_coil", 0.3),
    ("spark_gap", 0.25),
    ("plasma_globe", 1.0),
    ("capacitor_charge", 1.2),
    ("electric_motor", 0.5),
    ("transformer", 0.4),
    ("power_grid", 1.0),
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
