"""The Menger sponge — an exact-SDF fractal, ray-marched.

Sphere-traces `procedural.menger_de` (Quilez's *exact* signed distance: a box
with a drilled cross carved at every level). Because the distance is exact, the
sponge takes crisp edges and clean hard-ish shadows — the sharpest of the fractal
family. Warm sandstone shading from the orbit trap, soft shadows + AO, dark sky,
host post. The recursion **depth grows** over time (1 → 4 levels) so the holes
drill in, and the sponge slowly turns. `--quality` scales the march counts. See
``docs/research/14-kifs-fractals.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.fractal import menger_de
from ..scene import Scene

_TWO_PI = wp.constant(6.2831853)


@wp.func
def _roty(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


@wp.func
def _de(p: wp.vec3, iters: int) -> float:
    return menger_de(p, iters)[0]


@wp.func
def _normal(p: wp.vec3, iters: int) -> wp.vec3:
    e = 0.0009
    dx = _de(p + wp.vec3(e, 0.0, 0.0), iters) - _de(p - wp.vec3(e, 0.0, 0.0), iters)
    dy = _de(p + wp.vec3(0.0, e, 0.0), iters) - _de(p - wp.vec3(0.0, e, 0.0), iters)
    dz = _de(p + wp.vec3(0.0, 0.0, e), iters) - _de(p - wp.vec3(0.0, 0.0, e), iters)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _palette(t: float) -> wp.vec3:
    # warm sandstone shells keyed on the orbit trap
    a = wp.vec3(0.55, 0.45, 0.35)
    b = wp.vec3(0.42, 0.36, 0.28)
    c = wp.vec3(1.0, 1.0, 1.0)
    d = wp.vec3(0.0, 0.12, 0.28)
    ph = (c * t + d) * _TWO_PI
    return a + wp.cw_mul(b, wp.vec3(wp.cos(ph[0]), wp.cos(ph[1]), wp.cos(ph[2])))


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, iters: int, steps: int) -> float:
    res = float(1.0)
    t = float(0.015)
    for _ in range(steps):
        h = _de(ro + rd * t, iters)
        if h < 0.0006:
            return 0.0
        res = wp.min(res, 12.0 * h / t)
        t += wp.clamp(h, 0.006, 0.15)
        if t > 5.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, iters: int) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.008 + 0.04 * float(k)
        d = _de(p + n * hr, iters)
        occ += (hr - d) * sca
        sca *= 0.8
    return wp.clamp(1.0 - 2.4 * occ, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  iters: int, spin: float, march_steps: int, shadow_steps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.0)
    hit = int(0)
    trap = float(0.0)
    for _ in range(march_steps):
        p = _roty(ro + rd * t, spin)
        v4 = menger_de(p, iters)
        d = v4[0]
        if d < 0.0004 * t + 0.0003:
            hit = 1
            trap = v4[1]
            break
        t += d * 0.9
        if t > 8.0:
            break

    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.03, 0.035, 0.05) * (1.0 - up) + wp.vec3(0.06, 0.07, 0.10) * up

    if hit == 1:
        p = _roty(ro + rd * t, spin)
        n = _normal(p, iters)
        base = _palette(trap * 1.4 + 0.15)
        ndl = wp.max(wp.dot(n, sun), 0.0)
        sh = _soft_shadow(p + n * 0.003, sun, iters, shadow_steps)
        ao = _ao(p, n, iters)
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        col = wp.cw_mul(base, wp.vec3(0.16, 0.17, 0.22) * ao            # ambient
                        + wp.vec3(1.0, 0.93, 0.78) * (ndl * sh))         # warm sun
        col = col + base * (rim * 0.25)                                  # edge sheen

    img[i, j] = col


def _tier_steps(name):
    return {"low": (110, 24), "medium": (170, 34), "high": (230, 46),
            "ultra": (320, 64)}.get(name, (170, 34))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _tier_steps(tier.name)
    # recursion depth drills in over time: 1 -> 4 levels
    depth = 1 + int(3.0 * (0.5 + 0.5 * math.sin(time * 0.3 - 1.5)))
    spin = time * 0.14 + float(mouse[0]) * 0.01
    az = 0.7
    el = 0.35 + float(mouse[1]) * 0.004
    dist = 3.4
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=42.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.55, 0.6, 0.35))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(depth), float(spin), int(ms), int(ss),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.3, radius=r, passes=2)
    out = post.tonemap(hdr, mode="aces", exposure=1.05)
    return post.vignette(out, 0.3)


SCENE = Scene(
    name="menger",
    description="Distance-estimated Menger sponge (Quilez exact SDF) — crisp "
                "drilled-cube fractal, warm sandstone, depth drills in 1->4.",
    renderer=_render,
)
