"""Kaleidoscopic IFS — fractal architecture, ray-marched.

Sphere-traces `procedural.kifs_de` (octant fold + rotation + Sierpinski folds +
scale about a corner). The rotation in the loop breaks the self-similar copies off
their grid so they spiral into columns and arches — endless fractal "temple."
Warm-stone orbit-trap colour, a soft glow, soft shadows + AO, dark sky, host post.
The **fold angle rotates** over time so the architecture continually reforms.
`--quality` scales the march / iteration counts. See
``docs/research/14-kifs-fractals.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.fractal import kifs_de
from ..scene import Scene

_TWO_PI = wp.constant(6.2831853)
_SCALE = wp.constant(1.85)


@wp.func
def _roty(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


@wp.func
def _de(p: wp.vec3, angle: float, iters: int) -> float:
    return kifs_de(p, _SCALE, angle, iters)[0]


@wp.func
def _normal(p: wp.vec3, angle: float, iters: int) -> wp.vec3:
    e = 0.0012
    dx = _de(p + wp.vec3(e, 0.0, 0.0), angle, iters) - _de(p - wp.vec3(e, 0.0, 0.0), angle, iters)
    dy = _de(p + wp.vec3(0.0, e, 0.0), angle, iters) - _de(p - wp.vec3(0.0, e, 0.0), angle, iters)
    dz = _de(p + wp.vec3(0.0, 0.0, e), angle, iters) - _de(p - wp.vec3(0.0, 0.0, e), angle, iters)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _palette(t: float) -> wp.vec3:
    # warm temple stone with cool shadow shells
    a = wp.vec3(0.5, 0.42, 0.32)
    b = wp.vec3(0.45, 0.4, 0.32)
    c = wp.vec3(1.0, 1.0, 1.0)
    d = wp.vec3(0.1, 0.2, 0.4)
    ph = (c * t + d) * _TWO_PI
    return a + wp.cw_mul(b, wp.vec3(wp.cos(ph[0]), wp.cos(ph[1]), wp.cos(ph[2])))


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, angle: float, iters: int, steps: int) -> float:
    res = float(1.0)
    t = float(0.02)
    for _ in range(steps):
        h = _de(ro + rd * t, angle, iters)
        if h < 0.0008:
            return 0.0
        res = wp.min(res, 10.0 * h / t)
        t += wp.clamp(h, 0.01, 0.25)
        if t > 8.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, angle: float, iters: int) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _de(p + n * hr, angle, iters)
        occ += (hr - d) * sca
        sca *= 0.8
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  angle: float, iters: int, spin: float, march_steps: int,
                  shadow_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.0)
    glow = float(0.0)
    hit = int(0)
    trap = float(0.0)
    for _ in range(march_steps):
        p = _roty(ro + rd * t, spin)
        v4 = kifs_de(p, _SCALE, angle, iters)
        d = v4[0]
        glow = glow + wp.exp(-d * 34.0)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            trap = v4[1]
            break
        t += d * 0.85
        if t > 9.0:
            break

    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.02, 0.03, 0.05) * (1.0 - up) + wp.vec3(0.05, 0.06, 0.09) * up

    if hit == 1:
        p = _roty(ro + rd * t, spin)
        n = _normal(p, angle, iters)
        base = _palette(trap * 1.2 + 0.15)
        ndl = wp.max(wp.dot(n, sun), 0.0)
        sh = _soft_shadow(p + n * 0.005, sun, angle, iters, shadow_steps)
        ao = _ao(p, n, angle, iters)
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.5)
        col = wp.cw_mul(base, wp.vec3(0.15, 0.16, 0.2) * ao
                        + wp.vec3(1.0, 0.92, 0.78) * (ndl * sh))
        col = col + base * (rim * 0.35)

    col = col + wp.vec3(0.4, 0.5, 0.75) * (glow * 0.03)          # halo
    img[i, j] = col


def _tier_steps(name):
    return {"low": (100, 24, 12), "medium": (150, 34, 15), "high": (210, 46, 18),
            "ultra": (300, 66, 24)}.get(name, (150, 34, 15))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss, iters = _tier_steps(tier.name)
    # the fold angle sweeps slowly — the architecture reforms
    angle = 0.52 + 0.16 * math.sin(time * 0.18) + float(mouse[1]) * 0.003
    spin = time * 0.1 + float(mouse[0]) * 0.01
    az = 0.7
    el = 0.32
    dist = 3.9
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=44.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.6, 0.55, 0.4))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(angle), int(iters), float(spin),
                      int(ms), int(ss), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.013))
    hdr = post.bloom(hdr, threshold=1.05, strength=0.4, radius=r, passes=3)
    out = post.tonemap(hdr, mode="aces", exposure=1.1)
    return post.vignette(out, 0.3)


SCENE = Scene(
    name="kifs_temple",
    description="Distance-estimated kaleidoscopic IFS (Knighty) — fractal "
                "architecture of columns + arches; the fold angle reforms it.",
    renderer=_render,
)
