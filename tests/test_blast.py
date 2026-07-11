"""Tests for warp_shaders.blast.physics — nuclear scaling laws.

Run: `python -m tests.test_blast` (or under pytest). Checks the laws reproduce
the measured Tsar Bomba anchors and scale correctly to the Super Tsar.
"""

import numpy as np

from warp_shaders.blast import physics as P


def test_tsar_anchors():
    # calibrated to the measured 50 Mt Tsar Bomba test (within ~10%)
    assert abs(P.fireball_radius(P.TSAR_KT) - 3500.0) < 350.0      # ~3.5 km
    assert abs(P.thermal_radius(P.TSAR_KT) - 100000.0) < 10000.0   # ~100 km
    assert 33000.0 < P.destruction_radius(P.TSAR_KT) < 42000.0     # ~35 km (5 psi)
    # ordering of the overpressure rings
    assert P.severe_radius(P.TSAR_KT) < P.destruction_radius(P.TSAR_KT)
    assert P.destruction_radius(P.TSAR_KT) < P.light_radius(P.TSAR_KT)


def test_super_tsar_scaling():
    # x10 yield: fireball x 10^0.4 ~= 2.5, blast x 10^(1/3) ~= 2.15
    fr = P.fireball_radius(P.SUPER_TSAR_KT) / P.fireball_radius(P.TSAR_KT)
    dr = P.destruction_radius(P.SUPER_TSAR_KT) / P.destruction_radius(P.TSAR_KT)
    assert abs(fr - 10.0 ** 0.4) < 0.02
    assert abs(dr - 10.0 ** (1.0 / 3.0)) < 0.02


def test_overpressure_rings():
    # canonical anchors: 5 psi ~ destruction, 20 psi tighter, 1 psi wider
    assert P.overpressure_radius(P.TSAR_KT, 20.0) < P.overpressure_radius(P.TSAR_KT, 5.0)
    assert P.overpressure_radius(P.TSAR_KT, 5.0) < P.overpressure_radius(P.TSAR_KT, 1.0)


def test_shock_front_monotonic():
    t = np.linspace(0.1, 20.0, 40)
    r = P.shock_radius(t, P.TSAR_KT)
    assert np.all(np.diff(r) > 0.0)               # front always advances
    assert np.all(np.diff(np.diff(r)) < 1.0)      # decelerates (t^2/5)


def test_mushroom_rise():
    t = np.linspace(0.0, 120.0, 60)
    h = P.mushroom_height(t, P.TSAR_KT)
    assert np.all(np.diff(h) >= 0.0)              # monotone rise
    assert h[0] < 1.0                             # starts at the ground
    assert 55000.0 < h[-1] < 70000.0             # saturates near the ~67 km ceiling


def test_fireball_temp_cools():
    tn = np.linspace(0.0, 1.0, 20)
    T = P.fireball_temp(tn)
    assert np.all(np.diff(T) < 0.0)               # monotone cooling
    assert 1400.0 < T[-1] < 3000.0                # dull red at the end
    assert 25000.0 < T[0] < 32000.0               # blue-white at the start


def test_debris_shell_linear():
    t = np.array([1.0, 2.0, 4.0])
    r = P.debris_shell_radius(t, P.SUPER_TSAR_KT)
    assert np.all(np.diff(r) > 0.0)
    # ballistic: radius is linear in time
    assert abs((r[1] - r[0]) - (r[2] - r[1]) / 2.0) < 1.0


if __name__ == "__main__":
    test_tsar_anchors()
    print("  Tsar anchors (fireball 3.5 km, thermal 100 km, destruction 35 km): OK")
    print("   ", P.TSAR.summary())
    print("   ", P.SUPER_TSAR.summary())
    test_super_tsar_scaling()
    print("  Super Tsar scaling (fireball x2.5, blast x2.15): OK")
    test_overpressure_rings()
    print("  overpressure rings ordered (20 < 5 < 1 psi): OK")
    test_shock_front_monotonic()
    print("  Sedov shock front advances + decelerates: OK")
    test_mushroom_rise()
    print("  mushroom rise monotone -> ~67 km ceiling: OK")
    test_fireball_temp_cools()
    print("  fireball blackbody cools 30000 K -> ~1500 K: OK")
    test_debris_shell_linear()
    print("  vacuum debris shell ballistic (linear in t): OK")
    print("ALL PASSED")
