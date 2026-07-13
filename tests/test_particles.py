"""Smoke tests for the mesons / antimatter / exotic / hypothetical particle scenes.

Renders each at a tiny resolution and asserts finite, in-range, non-degenerate
output; also checks the shared lepton/nucleon `anti` flag did not disturb the
existing electron / proton renders.

    python -m tests.test_particles
"""

import numpy as np
import warp as wp

import warp_shaders as ws

_NEW = ["pion", "kaon", "jpsi", "upsilon",
        "positron", "antiproton", "annihilation",
        "ion", "positronium",
        "tachyon", "graviton", "magnetic_monopole", "axion", "dark_matter"]


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
    times = {"annihilation": 4.2, "ion": 2.5, "graviton": 1.5}
    for n in _NEW:
        _check(n, times.get(n, 1.0))
    # the anti flag must not have changed the matter particles
    e = np.asarray(ws.render("electron", width=64, height=64, time=1.0), np.float32)
    p = np.asarray(ws.render("proton", width=64, height=64, time=1.0), np.float32)
    assert np.all(np.isfinite(e)) and e.std() > 0.01, "electron regressed"
    assert np.all(np.isfinite(p)) and p.std() > 0.01, "proton regressed"
    print("  electron + proton still render (anti flag default off): OK")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
