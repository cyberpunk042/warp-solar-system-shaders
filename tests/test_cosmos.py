"""Tests for the configurable solar system (:mod:`warp_shaders.cosmos`).

Run: `python -m tests.test_cosmos` (or under pytest). Verifies the orbital
mechanics (Kepler + N-body), the remnant/collapse physics, the star config, that
a system renders finite, and that every system scene is registered.
"""

import math

import numpy as np
import warp as wp

from warp_shaders.cosmos import bodies as B
from warp_shaders.cosmos import orbits as O
from warp_shaders.cosmos import presets
from warp_shaders.cosmos.system import render_system

wp.init()


def test_orbit_circular_constant_radius():
    orb = O.Orbit(a=5.0, e=0.0, period=6.0)
    rs = [np.linalg.norm(O.orbit_position(orb, t)) for t in np.linspace(0, 6, 24)]
    assert max(rs) - min(rs) < 1e-3
    assert abs(np.mean(rs) - 5.0) < 1e-3


def test_orbit_eccentric_range():
    orb = O.Orbit(a=5.0, e=0.5, period=6.0)
    rs = [np.linalg.norm(O.orbit_position(orb, t)) for t in np.linspace(0, 6, 400)]
    assert abs(min(rs) - 2.5) < 0.1        # perihelion a(1-e)
    assert abs(max(rs) - 7.5) < 0.1        # aphelion a(1+e)


def test_orbit_inclination_and_period():
    orb = O.Orbit(a=5.0, e=0.0, incl=0.6, period=6.0)
    ys = [abs(O.orbit_position(orb, t)[1]) for t in np.linspace(0, 6, 50)]
    assert max(ys) > 1.0                    # lifts out of the XZ plane
    flat = O.Orbit(a=5.0, e=0.0, period=6.0)
    assert np.allclose(O.orbit_position(flat, 0.0),
                       O.orbit_position(flat, 6.0), atol=1e-3)   # period closes


def test_kepler_solver():
    for e in (0.0, 0.3, 0.7, 0.9):
        for M in np.linspace(-3.0, 3.0, 20):
            E = O.solve_kepler(float(M), e)
            # E - e sin E == M (mod 2pi)
            resid = (E - e * math.sin(E)) - ((M + math.pi) % (2 * math.pi) - math.pi)
            assert abs(resid) < 1e-4


def test_nbody_two_body_bound():
    pos = np.array([[-2, 0, 0], [2, 0, 0]], np.float32)
    v = O.circular_speed(1.0, 4.0) * 0.5
    vel = np.array([[0, 0, v], [0, 0, -v]], np.float32)
    mass = np.array([1.0, 1.0], np.float32)
    p, vv = pos.copy(), vel.copy()
    for _ in range(300):
        p, vv = O.nbody_step(p, vv, mass, dt=0.02)
    assert np.linalg.norm(p[0] - p[1]) < 10.0        # stays bound
    assert np.all(np.isfinite(p))


def test_remnant_thresholds():
    assert O.remnant_type(2.0) == "sun"
    assert O.remnant_type(30.0) == "neutron"
    assert O.remnant_type(50.0) == "black_hole"
    assert O.is_collapse(20.0, 30.0) == "neutron"
    assert O.is_collapse(30.0, 50.0) == "black_hole"
    assert O.is_collapse(2.0, 6.0) == ""             # star -> star, no collapse


def test_star_config():
    for kind in (B.SUN, B.NEUTRON, B.WHITE_DWARF, B.BLACK_HOLE):
        cfg = B.make_star(kind=kind, radius=0.7, temp=0.6)
        assert cfg.kind == kind and cfg.radius == 0.7


def test_presets_build():
    for n in presets.names():
        sc = presets.get(n)
        assert len(sc.stars) >= 1


def test_system_renders_finite():
    sc = presets.get("first")
    img = render_system(sc, 96, 64, time=1.0, device="cpu")
    assert img.shape == (64, 96, 3)
    assert np.all(np.isfinite(img)) and img.min() >= 0.0
    # the planet is somewhere on screen -> not a pure-black frame
    assert img.max() > 0.3


def test_scenes_registered():
    import warp_shaders.scenes.solar_system as ss
    import warp_shaders.scenes.ss_collapse as sc
    names = {s.name for s in ss.SCENES} | {sc.SCENE.name}
    for want in ("solar_system", "ss_binary", "ss_trinary", "ss_blackhole",
                 "ss_nebula", "ss_collapse"):
        assert want in names


if __name__ == "__main__":
    test_orbit_circular_constant_radius()
    print("  circular orbit constant radius: OK")
    test_orbit_eccentric_range()
    print("  eccentric orbit r in [a(1-e), a(1+e)]: OK")
    test_orbit_inclination_and_period()
    print("  inclination + period closure: OK")
    test_kepler_solver()
    print("  Kepler equation solver: OK")
    test_nbody_two_body_bound()
    print("  N-body 2-body bound: OK")
    test_remnant_thresholds()
    print("  remnant thresholds + collapse: OK")
    test_star_config()
    print("  star config: OK")
    test_presets_build()
    print("  presets build:", len(presets.names()), "OK")
    test_system_renders_finite()
    print("  system renders finite: OK")
    test_scenes_registered()
    print("  system scenes registered: OK")
    print("ALL PASSED")
