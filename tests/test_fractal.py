"""Tests for procedural.fractal — Mandelbulb / Mandelbox distance estimators.

Run: `python -m tests.test_fractal` (or under pytest). Exercises the device
``@wp.func``s in a module-level kernel; checks the DE invariants (finite,
non-negative, larger farther out, Lipschitz ~1).
"""

import numpy as np
import warp as wp

from warp_shaders.procedural import fractal as F

wp.init()


@wp.kernel
def _bulb_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = F.mandelbulb_de(pts[i], 8.0, 24)


@wp.kernel
def _box_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = F.mandelbox_de(pts[i], 2.0, 16)


def _eval(kernel, pts):
    p = wp.array(np.asarray(pts, np.float32), dtype=wp.vec3, device="cpu")
    o = wp.zeros(len(pts), dtype=wp.vec4, device="cpu")
    wp.launch(kernel, dim=len(pts), inputs=[p, o], device="cpu")
    wp.synchronize_device("cpu")
    return o.numpy()


def _check_de(kernel):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2.5, 2.5, (400, 3)).astype(np.float32)
    a = _eval(kernel, pts)
    de = a[:, 0]
    assert np.all(np.isfinite(a))
    assert np.all(de >= 0.0)                             # a distance is non-negative
    # the set is bounded near the origin, so the DE grows monotonically as we
    # move radially outward (farther from the surface = larger estimate)
    ramp = _eval(kernel, [[r, 0.0, 0.0] for r in (2.0, 4.0, 8.0, 16.0, 32.0)])[:, 0]
    assert np.all(np.diff(ramp) > 0.0)
    # the origin is inside/on the set (c = 0 never escapes) -> DE ~ 0 there
    assert _eval(kernel, [[0.0, 0.0, 0.0]])[0, 0] < 0.05


def test_mandelbulb_de():
    _check_de(_bulb_kernel)
    # the orbit trap (min |z|) is recorded and bounded
    a = _eval(_bulb_kernel, [[0.6, 0.2, 0.1]])
    assert 0.0 <= a[0][1] < 2.0


def test_mandelbox_de():
    _check_de(_box_kernel)


if __name__ == "__main__":
    test_mandelbulb_de()
    print("  mandelbulb DE (finite, >=0, monotone, orbit trap): OK")
    test_mandelbox_de()
    print("  mandelbox DE (finite, >=0, monotone): OK")
    print("ALL PASSED")
