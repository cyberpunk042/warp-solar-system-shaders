"""Smoke test for the waves & resonance strand.

Renders each scene at a tiny resolution and asserts finite, in-range, non-degenerate output;
also checks the wave solver is stable and the Bessel approximation is accurate.

    python -m tests.test_waves
"""

import numpy as np
import warp as wp

import warp_shaders as ws
from warp_shaders.scenes.standing_membrane import _j0, _j1, _jm
from warp_shaders.sim.wave import WaveField

_SCENES = ["chladni", "ripple_tank", "standing_membrane", "double_slit"]


def _check_scene(name):
    img = np.asarray(ws.render(name, width=96, height=72, time=0.0), np.float32)
    assert img.shape == (72, 96, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.05, f"{name}: essentially black"
    assert img.std() > 0.01, f"{name}: flat fill"
    print(f"  {name}: OK  (max {img.max():.3f} std {img.std():.3f})")


def _check_bessel():
    # J0(0)=1, J1(0)=0; known zeros J0(2.4048)=0, J1(3.8317)=0
    assert abs(_j0(np.array([0.0]))[0] - 1.0) < 1e-4
    assert abs(_j1(np.array([0.0]))[0]) < 1e-4
    assert abs(_j0(np.array([2.4048]))[0]) < 2e-3, "J0 zero wrong"
    assert abs(_j1(np.array([3.8317]))[0]) < 2e-3, "J1 zero wrong"
    # recurrence J2 should match a known value J2(5.0) ≈ 0.04657
    assert abs(_jm(2, np.array([5.0]))[0] - 0.04657) < 5e-3, "J2 recurrence wrong"
    print("  bessel: J0/J1/J2 accurate at known points + zeros")


def _check_wave_stability():
    f = WaveField(n=64, c=0.5)
    f.add_source(0.5, 0.5, amp=1.0, omega=0.4)
    f.run(120)
    assert np.all(np.isfinite(f.u)), "wave: non-finite"
    assert np.abs(f.u).max() < 10.0, "wave: blew up (Courant violated?)"
    assert np.abs(f.u).max() > 1e-4, "wave: source produced nothing"
    print("  wave solver: finite + stable after 120 steps")


def main():
    wp.init()
    _check_bessel()
    _check_wave_stability()
    for n in _SCENES:
        _check_scene(n)
    print(f"ALL PASSED ({len(_SCENES)} scenes + bessel + wave)")


if __name__ == "__main__":
    main()
