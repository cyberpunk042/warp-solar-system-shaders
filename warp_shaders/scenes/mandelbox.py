"""The Mandelbox — Lowe's fold fractal, distance-estimated and ray-marched.

Sphere-traces `procedural.mandelbox_de` (box-fold + sphere-fold + scale) into its
endless nested architecture of boxes and spheres. Orbit-trap colour with a cool
metallic palette, soft shadows + AO from the DE, a faint edge glow, dark sky. The
box slowly turns. `--quality` scales the march / iteration counts. See
``docs/research/13-3d-fractals.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.fractal import mandelbox_de
from ..scene import Scene

_TWO_PI = wp.constant(6.2831853)
_SCALE = wp.constant(-1.5)     # the iconic crisp Mandelbox (Lowe's original)


@wp.func
def _roty(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


@wp.func
def _de(p: wp.vec3, iters: int) -> float:
    return mandelbox_de(p, _SCALE, iters)[0]


@wp.func
def _normal(p: wp.vec3, iters: int) -> wp.vec3:
    e = 0.0015
    dx = _de(p + wp.vec3(e, 0.0, 0.0), iters) - _de(p - wp.vec3(e, 0.0, 0.0), iters)
    dy = _de(p + wp.vec3(0.0, e, 0.0), iters) - _de(p - wp.vec3(0.0, e, 0.0), iters)
    dz = _de(p + wp.vec3(0.0, 0.0, e), iters) - _de(p - wp.vec3(0.0, 0.0, e), iters)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _palette(t: float) -> wp.vec3:
    a = wp.vec3(0.36, 0.40, 0.46)
    b = wp.vec3(0.34, 0.36, 0.40)
    c = wp.vec3(1.0, 1.0, 1.0)
    d = wp.vec3(0.30, 0.45, 0.65)
    ph = (c * t + d) * _TWO_PI
    return a + wp.cw_mul(b, wp.vec3(wp.cos(ph[0]), wp.cos(ph[1]), wp.cos(ph[2])))


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, iters: int, steps: int) -> float:
    res = float(1.0)
    t = float(0.03)
    for _ in range(steps):
        h = _de(ro + rd * t, iters)
        if h < 0.001:
            return 0.0
        res = wp.min(res, 10.0 * h / t)
        t += wp.clamp(h, 0.02, 0.4)
        if t > 12.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, iters: int) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.02 + 0.10 * float(k)
        d = _de(p + n * hr, iters)
        occ += (hr - d) * sca
        sca *= 0.8
    return wp.clamp(1.0 - 1.8 * occ, 0.0, 1.0)


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
    glow = float(0.0)
    hit = int(0)
    trap = float(0.0)
    for _ in range(march_steps):
        p = _roty(ro + rd * t, spin)
        v4 = mandelbox_de(p, _SCALE, iters)
        d = v4[0]
        glow = glow + wp.exp(-d * 55.0)                 # tight near-surface halo only
        if d < 0.0006 * t + 0.0004:
            hit = 1
            trap = v4[1]
            break
        t += d * 0.85
        if t > 16.0:
            break

    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.02, 0.03, 0.045) * (1.0 - up) + wp.vec3(0.03, 0.05, 0.08) * up

    if hit == 1:
        p = _roty(ro + rd * t, spin)
        n = _normal(p, iters)
        base = _palette(trap * 0.8 + 0.2)
        ndl = wp.max(wp.dot(n, sun), 0.0)
        sh = _soft_shadow(p + n * 0.006, sun, iters, shadow_steps)
        ao = _ao(p, n, iters)
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.0)
        col = wp.cw_mul(base, wp.vec3(0.12, 0.14, 0.18) * ao
                        + wp.vec3(1.0, 0.95, 0.86) * (ndl * sh))
        col = col + base * (rim * 0.5)                     # metallic edge sheen

    col = col + wp.vec3(0.28, 0.42, 0.7) * (glow * 0.012)
    img[i, j] = col


def _tier_steps(name):
    return {"low": (100, 26, 10), "medium": (150, 40, 13), "high": (200, 54, 16),
            "ultra": (280, 76, 20)}.get(name, (150, 40, 13))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss, iters = _tier_steps(tier.name)
    spin = time * 0.12 + float(mouse[0]) * 0.01
    az = 0.5
    el = 0.45 + float(mouse[1]) * 0.004
    dist = 6.2
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=40.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.5, 0.6, 0.4))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(iters), float(spin), int(ms), int(ss),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.014))
    hdr = post.bloom(hdr, threshold=1.15, strength=0.4, radius=r, passes=3)
    out = post.tonemap(hdr, mode="aces", exposure=1.08)
    return post.vignette(out, 0.3)


SCENE = Scene(
    name="mandelbox",
    description="Distance-estimated Mandelbox fractal (Lowe box-fold + sphere-fold, "
                "scale -1.5) — the iconic ringed cube, orbit-trap iridescent colour.",
    renderer=_render,
)
