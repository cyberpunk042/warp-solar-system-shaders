"""The Sierpinski tetrahedron — the simplest 3D KIFS, ray-marched.

Sphere-traces `procedural.sierpinski_de` (three plane folds against the tetra
mirror planes, then scale ×2 about a corner). A cool crystalline palette from the
orbit trap, a soft glow, soft shadows + AO, dark sky, host post. The **fold count
grows** over time so the gasket subdivides in, and it slowly spins. `--quality`
scales the march / iteration counts. See ``docs/research/14-kifs-fractals.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.fractal import sierpinski_de
from ..scene import Scene

_TWO_PI = wp.constant(6.2831853)
_CTR = wp.constant(wp.vec3(0.33, 0.33, 0.33))     # empirical attractor centroid


@wp.func
def _roty(p: wp.vec3, a: float) -> wp.vec3:
    # rotate about the attractor centre so the spin keeps it framed
    q = p - _CTR
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * q[0] + s * q[2], q[1], -s * q[0] + c * q[2]) + _CTR


@wp.func
def _de(p: wp.vec3, iters: int) -> float:
    return sierpinski_de(p, iters)[0]


@wp.func
def _normal(p: wp.vec3, iters: int) -> wp.vec3:
    e = 0.0011
    dx = _de(p + wp.vec3(e, 0.0, 0.0), iters) - _de(p - wp.vec3(e, 0.0, 0.0), iters)
    dy = _de(p + wp.vec3(0.0, e, 0.0), iters) - _de(p - wp.vec3(0.0, e, 0.0), iters)
    dz = _de(p + wp.vec3(0.0, 0.0, e), iters) - _de(p - wp.vec3(0.0, 0.0, e), iters)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _palette(t: float) -> wp.vec3:
    # cool crystalline shells (jade / cyan / violet)
    a = wp.vec3(0.42, 0.5, 0.5)
    b = wp.vec3(0.4, 0.45, 0.45)
    c = wp.vec3(1.0, 1.0, 1.0)
    d = wp.vec3(0.5, 0.35, 0.15)
    ph = (c * t + d) * _TWO_PI
    return a + wp.cw_mul(b, wp.vec3(wp.cos(ph[0]), wp.cos(ph[1]), wp.cos(ph[2])))


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, iters: int, steps: int) -> float:
    res = float(1.0)
    t = float(0.02)
    for _ in range(steps):
        h = _de(ro + rd * t, iters)
        if h < 0.0008:
            return 0.0
        res = wp.min(res, 10.0 * h / t)
        t += wp.clamp(h, 0.01, 0.2)
        if t > 6.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, iters: int) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.01 + 0.05 * float(k)
        d = _de(p + n * hr, iters)
        occ += (hr - d) * sca
        sca *= 0.8
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


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
        v4 = sierpinski_de(p, iters)
        d = v4[0]
        glow = glow + wp.exp(-d * 40.0)
        if d < 0.0005 * t + 0.0004:
            hit = 1
            trap = v4[1]
            break
        t += d * 0.85
        if t > 7.0:
            break

    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.02, 0.03, 0.05) * (1.0 - up) + wp.vec3(0.04, 0.06, 0.09) * up

    if hit == 1:
        p = _roty(ro + rd * t, spin)
        n = _normal(p, iters)
        base = _palette(trap * 1.6 + 0.2)
        ndl = wp.max(wp.dot(n, sun), 0.0)
        sh = _soft_shadow(p + n * 0.004, sun, iters, shadow_steps)
        ao = _ao(p, n, iters)
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.5)
        col = wp.cw_mul(base, wp.vec3(0.13, 0.16, 0.2) * ao
                        + wp.vec3(0.95, 0.98, 1.0) * (ndl * sh))
        col = col + base * (rim * 0.4)

    col = col + wp.vec3(0.35, 0.55, 0.8) * (glow * 0.03)          # cool halo
    img[i, j] = col


def _tier_steps(name):
    return {"low": (100, 22, 12), "medium": (150, 32, 15), "high": (200, 44, 18),
            "ultra": (280, 64, 22)}.get(name, (150, 32, 15))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss, iters = _tier_steps(tier.name)
    spin = time * 0.16 + float(mouse[0]) * 0.01
    az = 1.15
    el = 0.4 + float(mouse[1]) * 0.004
    dist = 3.0
    tgt = (0.33, 0.33, 0.33)                    # empirical attractor centroid
    eye = (tgt[0] + dist * math.cos(el) * math.sin(az),
           tgt[1] + dist * math.sin(el),
           tgt[2] + dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, tgt, fov_deg=42.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.5, 0.6, 0.5))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(iters), float(spin), int(ms), int(ss),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.013))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    out = post.tonemap(hdr, mode="aces", exposure=1.1)
    return post.vignette(out, 0.3)


SCENE = Scene(
    name="sierpinski",
    description="Distance-estimated Sierpinski tetrahedron (plane folds + scale) "
                "— crystalline 3D gasket, orbit-trap colour, cool glow.",
    renderer=_render,
)
