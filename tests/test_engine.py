"""Tests for the engine core additions — colour science + ray intersection.

Run: `python -m tests.test_engine` (or under pytest). Exercises the device
`@wp.func`s inside real (module-level) kernels so Warp can compile them, plus
the host colour twins.
"""

import numpy as np
import warp as wp

from warp_shaders.engine import color as C
from warp_shaders.engine import intersect as I
from warp_shaders.engine import post as P
from warp_shaders.engine import shadow as SH
from warp_shaders.engine import sky as S

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
def _sky_kernel(out: wp.array2d(dtype=wp.vec3), n: int):
    i = wp.tid()
    a = float(i) / float(n) * 6.2831853
    rd = wp.vec3(wp.cos(a), 0.15 * wp.sin(a * 3.0), wp.sin(a))
    out[i, 0] = S.starfield(rd)
    out[i, 1] = S.milky_way(rd, wp.vec3(0.0, 1.0, 0.0), 0.3)


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


@wp.kernel
def _shadow_kernel(out: wp.array(dtype=wp.vec4)):
    o = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.vec3(0.0, 0.0, 1.0)
    # soft shadow: sphere squarely on the light ray -> full shadow (0);
    # sphere pushed aside -> lit (1); sphere just past the edge -> penumbra.
    blocked = SH.soft_shadow_sphere(o, fwd, wp.vec3(0.0, 0.0, 5.0), 1.0, 12.0)
    lit = SH.soft_shadow_sphere(o, fwd, wp.vec3(3.0, 0.0, 5.0), 1.0, 12.0)
    graze = SH.soft_shadow_sphere(o, fwd, wp.vec3(1.05, 0.0, 5.0), 1.0, 12.0)
    out[0] = wp.vec4(blocked, lit, graze, 0.0)
    # AO: a near sphere the normal faces -> occluded; normal facing away -> none;
    # a far sphere -> ~none.
    ao_near = SH.sphere_ao(o, fwd, wp.vec3(0.0, 0.0, 1.2), 1.0)
    ao_away = SH.sphere_ao(o, wp.vec3(0.0, 0.0, -1.0), wp.vec3(0.0, 0.0, 1.2), 1.0)
    ao_far = SH.sphere_ao(o, fwd, wp.vec3(0.0, 0.0, 50.0), 1.0)
    ao_horiz = SH.sphere_occlusion(o, wp.vec3(0.0, 1.0, 0.0),
                                   wp.vec3(2.0, 0.0, 2.0), 1.0)   # partial-horizon branch
    out[1] = wp.vec4(ao_near, ao_away, ao_far, ao_horiz)
    # atoms
    pen = SH.penumbra(1.0, 0.1, 1.0, 8.0)                        # min(1, 0.8) = 0.8
    g0 = SH.ground_contact_ao(0.0, 1.0)                          # 0
    g5 = SH.ground_contact_ao(5.0, 1.0)                          # ~0.993
    out[2] = wp.vec4(pen, g0, g5, 0.0)


def test_shadow_device():
    o = wp.zeros(3, dtype=wp.vec4, device="cpu")
    wp.launch(_shadow_kernel, dim=1, inputs=[o], device="cpu")
    wp.synchronize_device("cpu")
    a = o.numpy()
    assert a[0][0] < 1e-4                       # squarely blocked -> full shadow
    assert a[0][1] > 0.999                       # pushed aside -> lit
    assert 0.02 < a[0][2] < 0.98                 # just past the edge -> penumbra
    assert 0.0 <= a[1][0] < 0.8                  # near facing sphere is occluded
    assert a[1][1] > 0.999                        # normal facing away -> unoccluded
    assert a[1][2] > 0.98                         # far sphere barely occludes
    assert 0.0 <= a[1][3] <= 1.0                  # partial-horizon branch stays in range
    assert a[1][0] < a[1][2]                      # near occludes more than far
    assert abs(a[2][0] - 0.8) < 1e-4             # penumbra atom
    assert a[2][1] < 1e-4 and a[2][2] > 0.99     # contact AO: 0 at ground, ~1 high up


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


def test_sky_device():
    n = 256
    o = wp.zeros((n, 2), dtype=wp.vec3, device="cpu")
    wp.launch(_sky_kernel, dim=n, inputs=[o, n], device="cpu")
    wp.synchronize_device("cpu")
    a = o.numpy()
    assert np.all(np.isfinite(a)) and a.min() >= 0.0
    assert a[:, 0].max() > 0.0          # stars emit (sparse; brightness verified visually)
    assert a[:, 1].max() > 0.0          # milky-way band emits somewhere


def test_post_ops():
    rng = np.random.default_rng(0)
    hdr = (rng.random((32, 48, 3)).astype(np.float32) * 3.0)   # HDR-ish
    # exposure doubles per stop
    assert np.allclose(P.exposure(hdr, 1.0), hdr * 2.0, atol=1e-4)
    # auto-exposure brings geometric-mean luminance near the key
    ae = P.auto_exposure(hdr * 0.01, key=0.18)
    lum = 0.2126 * ae[..., 0] + 0.7152 * ae[..., 1] + 0.0722 * ae[..., 2]
    gm = float(np.exp(np.mean(np.log(np.maximum(lum, 1e-4)))))
    assert 0.08 < gm < 0.4
    disp = np.clip(hdr / 3.0, 0, 1)
    for op in (P.chromatic_aberration, P.sharpen):
        out = op(disp)
        assert out.shape == disp.shape and np.all(np.isfinite(out))
    # film grain is deterministic from seed, and actually changes pixels
    g1 = P.film_grain(disp, amount=0.05, seed=3)
    g2 = P.film_grain(disp, amount=0.05, seed=3)
    assert np.array_equal(g1, g2) and not np.allclose(g1, disp)
    # chromatic aberration shifts channels near the edges (centre ~unchanged)
    ca = P.chromatic_aberration(disp, amount=0.02)
    assert not np.allclose(ca[:, 0], disp[:, 0])


def test_looks():
    # the named-look registry + one-call grade
    names = P.looks()
    assert names[0] == "clean" and {"cinematic", "film", "dreamy", "crisp"} <= set(names)
    rng = np.random.default_rng(1)
    disp = rng.random((24, 40, 3)).astype(np.float32)
    # "clean" is a no-op (identity within clamp), graded looks change pixels
    assert np.allclose(P.apply_look(disp, "clean"), np.clip(disp, 0, 1))
    for look in ("cinematic", "film", "dreamy", "crisp"):
        out = P.apply_look(disp, look, seed=2)
        assert out.shape == disp.shape and np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not np.allclose(out, disp)
    # a params dict works just like a preset name
    d = P.apply_look(disp, {"vignette": 0.5})
    assert d.shape == disp.shape and np.all(np.isfinite(d))
    # grained looks are deterministic from the seed
    assert np.array_equal(P.apply_look(disp, "film", 7), P.apply_look(disp, "film", 7))


if __name__ == "__main__":
    test_color_host_anchors()
    print("  colour host anchors (blackbody + sRGB): OK")
    test_color_device()
    print("  colour device (kelvin/blackbody/srgb/luminance): OK")
    test_intersect_device()
    print("  intersect device (sphere/plane/disk/box): OK")
    test_sky_device()
    print("  sky device (starfield + milky way): OK")
    test_post_ops()
    print("  post ops (exposure/auto/CA/grain/sharpen): OK")
    test_looks()
    print("  looks (clean/cinematic/film/dreamy/crisp + params dict): OK")
    test_shadow_device()
    print("  shadow device (soft sphere shadow + analytic AO + atoms): OK")
    print("ALL PASSED")
