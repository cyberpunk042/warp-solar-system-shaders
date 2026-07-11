"""Tests for engine.raytrace — reflection / refraction device funcs.

Run: `python -m tests.test_raytrace` (or under pytest). Exercises the ``@wp.func``s
inside a module-level kernel so Warp can compile them.
"""

import math

import numpy as np
import warp as wp

from warp_shaders.engine import raytrace as RT

wp.init()

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


@wp.kernel
def _rt_kernel(out: wp.array(dtype=wp.vec4)):
    n = wp.vec3(0.0, 1.0, 0.0)
    # reflect a 45-degree down-ray off the floor -> 45-degree up-ray (y flips)
    i = wp.vec3(0.70710678, -0.70710678, 0.0)
    r = RT.reflect(i, n)
    out[0] = wp.vec4(r[0], r[1], r[2], 0.0)
    # refract straight down into glass (eta = 1/1.5): no bend at normal incidence
    rd = RT.refract(wp.vec3(0.0, -1.0, 0.0), n, 1.0 / 1.5)
    out[1] = wp.vec4(rd[0], rd[1], rd[2], 0.0)
    # refract at 45 deg entering glass: bends TOWARD the normal (x shrinks)
    rr = RT.refract(i, n, 1.0 / 1.5)
    out[2] = wp.vec4(rr[0], rr[1], rr[2], 0.0)
    # total internal reflection: glass->air (eta=1.5) at a grazing 60 deg -> TIR
    graze = wp.vec3(0.8660254, -0.5, 0.0)
    kk = RT.refract_k(graze, n, 1.5)
    tir = RT.refract(graze, n, 1.5)                    # falls back to reflect
    out[3] = wp.vec4(kk, tir[1], 0.0, 0.0)             # tir[1] > 0 (reflected up)
    # fresnel: head-on ~F0(1.5)=0.04, grazing -> ~1
    f_head = RT.fresnel_dielectric(1.0, 1.5)
    f_graze = RT.fresnel_dielectric(0.02, 1.5)
    out[4] = wp.vec4(f_head, f_graze, RT.schlick_f0(1.5), 0.0)


def test_raytrace_device():
    o = wp.zeros(5, dtype=wp.vec4, device="cpu")
    wp.launch(_rt_kernel, dim=1, inputs=[o], device="cpu")
    wp.synchronize_device("cpu")
    a = o.numpy()
    # reflect: (0.707,-0.707,0) -> (0.707,+0.707,0)
    assert abs(a[0][0] - _INV_SQRT2) < 1e-4 and abs(a[0][1] - _INV_SQRT2) < 1e-4
    # refract normal incidence: still (0,-1,0)
    assert abs(a[1][0]) < 1e-4 and abs(a[1][1] + 1.0) < 1e-4
    # refract 45deg into denser medium: transmitted ray is more vertical, so |x|
    # shrinks below the incident 0.707 and it still points downward
    assert 0.0 < a[2][0] < _INV_SQRT2 - 1e-3 and a[2][1] < 0.0
    # TIR: discriminant negative, fallback reflection points up
    assert a[3][0] < 0.0 and a[3][1] > 0.0
    # fresnel monotonic: head-on ~0.04, grazing ~1, F0 matches
    assert abs(a[4][0] - 0.04) < 0.01 and a[4][1] > 0.9
    assert abs(a[4][2] - 0.04) < 0.01                  # schlick_f0(1.5)


if __name__ == "__main__":
    test_raytrace_device()
    print("  raytrace device (reflect/refract/TIR/fresnel): OK")
    print("ALL PASSED")
