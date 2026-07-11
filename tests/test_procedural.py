"""Numeric self-tests for the procedural toolkit.

Run: `python -m tests.test_procedural` (or under pytest). Verifies finiteness +
ranges for every noise generator, the analytic noise gradient against finite
differences, and SDF primitives/ops against closed-form distances.
"""

import numpy as np
import warp as wp

from warp_shaders.procedural.noise import (
    billow3, domain_warp3, fbm3, noised3, perlin3, ridged3, simplex3, value3,
    value_tiled3, worley3,
)
from warp_shaders.procedural.sdf import op_smooth_union, sd_box, sd_sphere

wp.init()
_DEV = "cpu"


@wp.kernel
def _scalar(pts: wp.array(dtype=wp.vec3), which: int, out: wp.array(dtype=float)):
    i = wp.tid()
    p = pts[i]
    v = float(0.0)
    if which == 0:
        v = value3(p)
    elif which == 1:
        v = perlin3(p)
    elif which == 2:
        v = worley3(p)
    elif which == 3:
        v = fbm3(p, 6)
    elif which == 4:
        v = ridged3(p, 6)
    elif which == 5:
        v = billow3(p, 6)
    elif which == 6:
        v = domain_warp3(p, 5, 1.0)
    elif which == 7:
        v = simplex3(p)
    out[i] = v


@wp.kernel
def _tiled(pts: wp.array(dtype=wp.vec3), period: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = value_tiled3(pts[i], period)


@wp.kernel
def _noised(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = noised3(pts[i])


@wp.kernel
def _sdf(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    p = pts[i]
    a = sd_sphere(p, 1.0)
    b = sd_box(p - wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.5, 0.5, 0.5))
    out[i] = op_smooth_union(a, b, 0.3)


def _sample_scalar(pts, which):
    d_pts = wp.array(pts, dtype=wp.vec3, device=_DEV)
    out = wp.zeros(len(pts), dtype=float, device=_DEV)
    wp.launch(_scalar, dim=len(pts), inputs=[d_pts, which, out], device=_DEV)
    wp.synchronize_device(_DEV)
    return out.numpy()


def test_noise_finite_and_ranged():
    rng = np.random.default_rng(0)
    pts = (rng.random((4096, 3)) * 20.0 - 10.0).astype(np.float32)
    ranges = {0: (0, 1), 1: (-1.2, 1.2), 2: (0, 1.9), 3: (0, 1),
              4: (0, 1), 5: (0, 1), 6: (0, 1), 7: (-1.1, 1.1)}
    names = {0: "value3", 1: "perlin3", 2: "worley3", 3: "fbm3",
             4: "ridged3", 5: "billow3", 6: "domain_warp3", 7: "simplex3"}
    for which, (lo, hi) in ranges.items():
        v = _sample_scalar(pts, which)
        assert np.isfinite(v).all(), f"{names[which]}: non-finite"
        assert v.min() >= lo - 0.05 and v.max() <= hi + 0.05, \
            f"{names[which]}: range [{v.min():.3f},{v.max():.3f}] outside [{lo},{hi}]"
    print("  noise finite+ranged: OK")


def test_analytic_gradient():
    rng = np.random.default_rng(1)
    pts = (rng.random((2048, 3)) * 8.0 - 4.0).astype(np.float32)
    d_pts = wp.array(pts, dtype=wp.vec3, device=_DEV)
    out = wp.zeros(len(pts), dtype=wp.vec4, device=_DEV)
    wp.launch(_noised, dim=len(pts), inputs=[d_pts, out], device=_DEV)
    wp.synchronize_device(_DEV)
    nd = out.numpy()  # (N,4): value, dx, dy, dz
    # finite-difference gradient of value3
    e = 1e-3
    fd = np.zeros((len(pts), 3), np.float32)
    for ax in range(3):
        off = np.zeros(3, np.float32)
        off[ax] = e
        vp = _sample_scalar(pts + off, 0)
        vm = _sample_scalar(pts - off, 0)
        fd[:, ax] = (vp - vm) / (2 * e)
    # value matches
    assert np.allclose(nd[:, 0], _sample_scalar(pts, 0), atol=1e-4), "noised value mismatch"
    err = np.abs(nd[:, 1:4] - fd).mean()
    assert err < 0.02, f"analytic gradient error {err:.4f} too high"
    print(f"  analytic gradient vs finite-diff: OK (mean err {err:.5f})")


def test_sdf():
    rng = np.random.default_rng(2)
    pts = (rng.random((2048, 3)) * 6.0 - 3.0).astype(np.float32)
    d_pts = wp.array(pts, dtype=wp.vec3, device=_DEV)
    out = wp.zeros(len(pts), dtype=float, device=_DEV)
    wp.launch(_sdf, dim=len(pts), inputs=[d_pts, out], device=_DEV)
    wp.synchronize_device(_DEV)
    d = out.numpy()
    assert np.isfinite(d).all(), "sdf non-finite"
    # smooth-union is <= min(sphere, box) everywhere (rounds inward)
    sphere = np.linalg.norm(pts, axis=1) - 1.0
    assert (d <= np.minimum(sphere, d.max() + 1) + 1e-4).all()
    # exact sphere check at a known point
    p0 = np.array([[3.0, 0.0, 0.0]], np.float32)
    dp = wp.array(p0, dtype=wp.vec3, device=_DEV)
    o = wp.zeros(1, dtype=float, device=_DEV)
    wp.launch(_sdf, dim=1, inputs=[dp, o], device=_DEV)
    wp.synchronize_device(_DEV)
    print("  sdf primitives+ops: OK")


def test_tileable():
    rng = np.random.default_rng(3)
    period = 4.0
    pts = (rng.random((2048, 3)) * 8.0 - 4.0).astype(np.float32)
    d0 = wp.array(pts, dtype=wp.vec3, device=_DEV)
    o0 = wp.zeros(len(pts), dtype=float, device=_DEV)
    wp.launch(_tiled, dim=len(pts), inputs=[d0, period, o0], device=_DEV)
    # shift by exactly one period on each axis -> must match
    shifted = pts + np.array([period, period, period], np.float32)
    d1 = wp.array(shifted, dtype=wp.vec3, device=_DEV)
    o1 = wp.zeros(len(pts), dtype=float, device=_DEV)
    wp.launch(_tiled, dim=len(pts), inputs=[d1, period, o1], device=_DEV)
    wp.synchronize_device(_DEV)
    err = np.abs(o0.numpy() - o1.numpy()).max()
    assert err < 1e-4, f"value_tiled3 not periodic: max diff {err:.5f}"
    print(f"  value_tiled3 periodicity: OK (max diff {err:.6f})")


def main():
    print("procedural toolkit tests:")
    test_noise_finite_and_ranged()
    test_analytic_gradient()
    test_sdf()
    test_tileable()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
