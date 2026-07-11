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


@wp.kernel
def _sier_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = F.sierpinski_de(pts[i], 12)


@wp.kernel
def _menger_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = F.menger_de(pts[i], 4)


@wp.kernel
def _kifs_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = F.kifs_de(pts[i], 2.0, 0.5, 12)


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


def test_sierpinski_de():
    # unsigned fold DE: finite, non-negative, corner vertex on the set
    a = _eval(_sier_kernel, [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    assert np.all(np.isfinite(a))
    assert np.all(a[:, 0] >= 0.0)
    assert a[0, 0] < 0.1                                 # (1,1,1) is a tetra vertex
    assert a[1, 0] > a[0, 0]                             # farther point, larger DE
    ramp = _eval(_sier_kernel, [[r, r, r] for r in (1.5, 3.0, 6.0, 12.0)])[:, 0]
    assert np.all(np.diff(ramp) > 0.0)                  # grows radially outward


def test_menger_de():
    # exact signed distance: outside is large-positive, a cube corner is on it
    a = _eval(_menger_kernel, [[3.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert np.all(np.isfinite(a))
    assert a[0, 0] > 1.0                                 # well outside the unit box
    assert abs(a[1, 0]) < 0.2                            # (1,1,1) on the sponge surface


def test_kifs_de():
    pts = [[0.5, 0.3, 0.2], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    a = _eval(_kifs_kernel, pts)
    assert np.all(np.isfinite(a))
    assert np.all(a[:, 0] >= 0.0)
    assert 0.0 <= a[0][1] < 4.0                          # orbit trap recorded, bounded
    ramp = _eval(_kifs_kernel, [[r, r, r] for r in (2.0, 4.0, 8.0, 16.0)])[:, 0]
    assert np.all(np.diff(ramp) > 0.0)


if __name__ == "__main__":
    test_mandelbulb_de()
    print("  mandelbulb DE (finite, >=0, monotone, orbit trap): OK")
    test_mandelbox_de()
    print("  mandelbox DE (finite, >=0, monotone): OK")
    test_sierpinski_de()
    print("  sierpinski DE (finite, >=0, vertex on set, monotone): OK")
    test_menger_de()
    print("  menger DE (exact signed, outside large, corner on surface): OK")
    test_kifs_de()
    print("  kifs DE (finite, >=0, orbit trap, monotone): OK")
    print("ALL PASSED")
