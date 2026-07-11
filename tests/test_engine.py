"""Tests for the engine core additions — colour science + ray intersection.

Run: `python -m tests.test_engine` (or under pytest). Exercises the device
`@wp.func`s inside real (module-level) kernels so Warp can compile them, plus
the host colour twins.
"""

import numpy as np
import warp as wp

from warp_shaders.engine import color as C
from warp_shaders.engine import intersect as I

wp.init()


@wp.kernel
def _color_kernel(out: wp.array(dtype=wp.vec3)):
    out[0] = C.kelvin_to_rgb(2000.0)          # reddish
    out[1] = C.kelvin_to_rgb(20000.0)         # blueish
    out[2] = C.blackbody(0.05)                # cold (deep red)
    out[3] = C.blackbody(0.95)                # hot (blue-white)
    out[4] = C.linear_to_srgb(wp.vec3(0.5, 0.5, 0.5))
    lum = C.luminance(wp.vec3(1.0, 1.0, 1.0))
    out[5] = wp.vec3(lum, lum, lum)


@wp.kernel
def _isect_kernel(out: wp.array(dtype=wp.vec4)):
    ro = wp.vec3(0.0, 0.0, -5.0)
    rd = wp.vec3(0.0, 0.0, 1.0)
    hit = I.ray_sphere(ro, rd, wp.vec3(0.0, 0.0, 0.0), 1.0)   # hits at t=4,6
    out[0] = wp.vec4(hit[0], hit[1], 0.0, 0.0)
    miss = I.ray_sphere(ro, wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    out[1] = wp.vec4(miss[0], miss[1], 0.0, 0.0)              # miss -> (1e30,-1e30)
    t = I.sphere_t(ro, rd, wp.vec3(0.0, 0.0, 0.0), 1.0)       # 4.0
    tp = I.ray_plane(ro, rd, wp.vec3(0.0, 0.0, 2.0),
                     wp.vec3(0.0, 0.0, -1.0))                 # 7.0
    td = I.ray_disk(ro, rd, wp.vec3(0.0, 0.0, 2.0),
                    wp.vec3(0.0, 0.0, -1.0), 0.5)             # -1 (off-axis? no, on axis -> 7)
    out[2] = wp.vec4(t, tp, td, 0.0)
    box = I.ray_box(ro, rd, wp.vec3(-1.0, -1.0, -1.0), wp.vec3(1.0, 1.0, 1.0))
    out[3] = wp.vec4(box[0], box[1], 0.0, 0.0)               # (4,6)


def test_color_host_anchors():
    assert C.kelvin_to_rgb_np(2000)[0] > C.kelvin_to_rgb_np(2000)[2]
    assert C.kelvin_to_rgb_np(20000)[2] >= C.kelvin_to_rgb_np(20000)[0] * 0.9
    mid = C.kelvin_to_rgb_np(6600)
    assert mid[0] > 0.9 and mid[1] > 0.85
    ks = np.linspace(1500, 30000, 40)
    assert C.kelvin_to_rgb_np(ks)[-1, 2] > C.kelvin_to_rgb_np(ks)[0, 2]
    x = np.array([[0.2, 0.5, 0.8]], np.float32)
    assert np.allclose(C.srgb_to_linear_np(C.linear_to_srgb_np(x)), x, atol=1e-3)


def test_color_device():
    o = wp.zeros(6, dtype=wp.vec3, device="cpu")
    wp.launch(_color_kernel, dim=1, inputs=[o], device="cpu")
    wp.synchronize_device("cpu")
    a = o.numpy()
    assert a[0][0] > a[0][2]                    # 2000K red>blue
    assert a[1][2] >= a[1][0] * 0.9             # 20000K blue>=red
    assert a[2][0] > a[2][2]                    # cold blackbody red>blue
    assert a[3][2] > a[3][0] * 0.7              # hot blackbody blue rises
    assert abs(a[4][0] - 0.5 ** (1.0 / 2.2)) < 1e-3   # srgb of 0.5
    assert abs(a[5][0] - 1.0) < 1e-4            # luminance of white == 1


def test_intersect_device():
    o = wp.zeros(4, dtype=wp.vec4, device="cpu")
    wp.launch(_isect_kernel, dim=1, inputs=[o], device="cpu")
    wp.synchronize_device("cpu")
    a = o.numpy()
    assert abs(a[0][0] - 4.0) < 1e-4 and abs(a[0][1] - 6.0) < 1e-4   # sphere hit
    assert a[1][0] > 1e29 and a[1][1] < -1e29                        # miss sentinel
    assert abs(a[2][0] - 4.0) < 1e-4                                 # sphere_t
    assert abs(a[2][1] - 7.0) < 1e-4                                 # ray_plane
    assert abs(a[2][2] - 7.0) < 1e-4                                 # ray_disk (on axis)
    assert abs(a[3][0] - 4.0) < 1e-4 and abs(a[3][1] - 6.0) < 1e-4   # ray_box


if __name__ == "__main__":
    test_color_host_anchors()
    print("  colour host anchors (blackbody + sRGB): OK")
    test_color_device()
    print("  colour device (kelvin/blackbody/srgb/luminance): OK")
    test_intersect_device()
    print("  intersect device (sphere/plane/disk/box): OK")
    print("ALL PASSED")
