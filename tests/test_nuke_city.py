"""Tests for the nuke-on-a-city collapse model (`blast.render._city_blast_de`).

Run: `python -m tests.test_nuke_city`. Samples the collapse factor along a ray
out from ground zero and checks it grades correctly: total near GZ, zero beyond
the overpressure front, monotonically non-increasing in between.
"""

import numpy as np
import warp as wp

from warp_shaders.blast.render import _city_blast_de

wp.init()

_LOT = 15.0
_FRONT = 55.0
_DEST = 60.0
_SEV = 16.0


@wp.kernel
def _collapse_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    gz = wp.vec2(0.0, 0.0)
    out[i] = _city_blast_de(pts[i], _FRONT, _DEST, _SEV, gz, _LOT, 3.0)[1]


def _collapse_at(xs):
    pts = np.array([[x, 5.0, 0.0] for x in xs], np.float32)
    p = wp.array(pts, dtype=wp.vec3, device="cpu")
    o = wp.zeros(len(xs), dtype=wp.float32, device="cpu")
    wp.launch(_collapse_kernel, dim=len(xs), inputs=[p, o], device="cpu")
    wp.synchronize_device("cpu")
    return o.numpy()


def test_collapse_grades_with_distance():
    xs = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]     # lot centres out from GZ
    c = _collapse_at(xs)
    assert np.all(np.isfinite(c))
    assert np.all((c >= 0.0) & (c <= 1.0))             # collapse is a [0,1] factor
    assert c[0] > 0.9                                  # ground zero: flattened
    assert c[-1] < 0.05                                # far suburb, front not reached
    assert np.all(np.diff(c) <= 1e-4)                  # non-increasing outward


def test_front_gates_collapse():
    # with the front at GZ (nothing swept yet) the whole city is intact
    pts = np.array([[x, 5.0, 0.0] for x in [0.0, 30.0, 60.0]], np.float32)
    p = wp.array(pts, dtype=wp.vec3, device="cpu")
    o = wp.zeros(3, dtype=wp.float32, device="cpu")

    @wp.kernel
    def _k(pp: wp.array(dtype=wp.vec3), oo: wp.array(dtype=wp.float32)):
        i = wp.tid()
        oo[i] = _city_blast_de(pp[i], 0.0, _DEST, _SEV, wp.vec2(0.0, 0.0), _LOT, 3.0)[1]

    wp.launch(_k, dim=3, inputs=[p, o], device="cpu")
    wp.synchronize_device("cpu")
    assert np.all(o.numpy() < 0.05)                    # front at 0 -> nothing collapsed


if __name__ == "__main__":
    test_collapse_grades_with_distance()
    print("  collapse grades total->zero, non-increasing outward: OK")
    test_front_gates_collapse()
    print("  overpressure front gates collapse (front at GZ -> intact): OK")
    print("ALL PASSED")
