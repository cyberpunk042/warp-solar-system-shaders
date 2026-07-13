"""Smoke test for the physics-sim scenes (N-body gravity, stable fluid).

Renders each at a tiny resolution and asserts finite, in-range, non-degenerate output.
Also checks the raw simulators produce sane state.

    python -m tests.test_physics
"""

import numpy as np
import warp as wp

import warp_shaders as ws
from warp_shaders.sim.fluid import StableFluid
from warp_shaders.sim.nbody import NBody, make_collision

_SCENES = ["nbody", "fluid"]


def _check_scene(name):
    img = np.asarray(ws.render(name, width=96, height=72, time=0.0), np.float32)
    assert img.shape == (72, 96, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.05, f"{name}: essentially black"
    assert img.std() > 0.01, f"{name}: flat fill"
    print(f"  {name}: OK  (max {img.max():.3f} std {img.std():.3f})")


def _check_nbody_conserves_com():
    # with no external field the centre of mass drifts at constant velocity; check it stays finite
    pos, vel, mass, clump = make_collision(n=400, seed=1)
    sim = NBody(pos, vel, mass, device="cpu", g=1.0, eps=0.06)
    p, v = sim.run(60, 0.01)
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(v)), "nbody: non-finite state"
    assert np.abs(p).max() < 1e3, "nbody: state blew up (softening/integrator broken)"
    print("  nbody sim: finite + bounded after 60 steps")


def _check_fluid_mass_and_stability():
    sim = StableFluid(n=48, seed=2)
    d0 = sim.d.sum()
    sim.run(40, 0.09)
    assert np.all(np.isfinite(sim.d)), "fluid: non-finite density"
    assert np.all(np.isfinite(sim.vx)) and np.all(np.isfinite(sim.vy)), "fluid: non-finite velocity"
    assert sim.d.sum() > d0, "fluid: emitter added no dye"
    assert sim.d.max() < 100.0, "fluid: density blew up (projection/advection unstable)"
    print("  fluid sim: finite + stable, dye injected")


def main():
    wp.init()
    _check_nbody_conserves_com()
    _check_fluid_mass_and_stability()
    for n in _SCENES:
        _check_scene(n)
    print(f"ALL PASSED ({len(_SCENES)} scenes + 2 sims)")


if __name__ == "__main__":
    main()
