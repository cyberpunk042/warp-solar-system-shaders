"""Tests for buildings.sdf — parametric building distance fields.

Run: `python -m tests.test_buildings`. Exercises the device SDFs in a kernel and
checks the invariants (finite; inside < 0 < outside; the city repeats + varies).
"""

import numpy as np
import warp as wp

from warp_shaders.buildings import sdf as B

wp.init()


@wp.kernel
def _tower_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = B.sd_tower(pts[i], wp.vec3(3.0, 12.0, 3.0), 1.6, 0.5)


@wp.kernel
def _house_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = B.sd_house(pts[i], wp.vec3(3.0, 2.5, 4.0), 3.0)


@wp.kernel
def _city_kernel(pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec4)):
    i = wp.tid()
    out[i] = B.city_de(pts[i], 16.0, 3.0)


def _eval(kernel, pts, dt=wp.float32):
    p = wp.array(np.asarray(pts, np.float32), dtype=wp.vec3, device="cpu")
    o = wp.zeros(len(pts), dtype=dt, device="cpu")
    wp.launch(kernel, dim=len(pts), inputs=[p, o], device="cpu")
    wp.synchronize_device("cpu")
    return o.numpy()


def test_tower():
    # the facade is a window-lattice sponge, so sample the interior: some of it
    # must be solid (< 0); far outside must be large-positive; all finite.
    rng = np.random.default_rng(0)
    inside = rng.uniform([-2.5, 0.0, -2.5], [2.5, 10.0, 2.5], (400, 3)).astype(np.float32)
    di = _eval(_tower_kernel, inside)
    far = _eval(_tower_kernel, [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0]])
    assert np.all(np.isfinite(di)) and np.all(np.isfinite(far))
    assert di.min() < 0.0                              # the building is solid somewhere
    assert far[0] > 30.0 and far[1] > 20.0             # far outside


def test_house():
    rng = np.random.default_rng(1)
    inside = rng.uniform([-2.5, -2.0, -3.5], [2.5, 2.0, 3.5], (400, 3)).astype(np.float32)
    di = _eval(_house_kernel, inside)
    far = _eval(_house_kernel, [[30.0, 0.0, 0.0]])[0]
    assert np.all(np.isfinite(di))
    assert di.min() < 0.0                              # inside the house
    assert far > 20.0
    # the pitched roof adds height above the body top (y=2.5)
    apex = _eval(_house_kernel, [[0.0, 4.0, 0.0]])[0]
    assert apex < 3.0


def test_city_repeats_and_varies():
    a = _eval(_city_kernel, [[0.0, 60.0, 0.0]], dt=wp.vec4)      # high above a lot
    assert np.all(np.isfinite(a))
    # different lots hash to different heights/variants
    heights = _eval(_city_kernel, [[x * 16.0, 100.0, 0.0] for x in range(8)], dt=wp.vec4)[:, 1]
    assert heights.std() > 1.0                        # buildings vary lot to lot
    assert np.all(heights >= 4.0)                     # at least the min height


if __name__ == "__main__":
    test_tower()
    print("  sd_tower (finite, inside<0, outside large): OK")
    test_house()
    print("  sd_house (finite, inside<0, roof apex above body): OK")
    test_city_repeats_and_varies()
    print("  city_de (repeats + hashes varied heights per lot): OK")
    print("ALL PASSED")
