"""The Mandelbulb — a distance-estimated 3D fractal, ray-marched.

Sphere-traces `procedural.mandelbulb_de` (White & Nylander triplex power), colours
the surface from the **orbit trap** with an IQ cosine palette, and adds a soft
**glow** from the ray's closest approach so the fractal floats in a luminous haze.
Soft shadows + AO from the DE, a dark-space sky, and the host post pipeline. The
**power morphs 2 → 8** over time (the lobes grow in) while the bulb slowly spins.
`--quality` scales the march / iteration counts. See
``docs/research/13-3d-fractals.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.fractal import mandelbulb_de
from ..scene import Scene

_TWO_PI = wp.constant(6.2831853)


@wp.func
def _roty(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


@wp.func
def _de(p: wp.vec3, power: float, iters: int) -> float:
    return mandelbulb_de(p, power, iters)[0]


@wp.func
def _normal(p: wp.vec3, power: float, iters: int) -> wp.vec3:
    e = 0.0012
    dx = _de(p + wp.vec3(e, 0.0, 0.0), power, iters) - _de(p - wp.vec3(e, 0.0, 0.0), power, iters)
    dy = _de(p + wp.vec3(0.0, e, 0.0), power, iters) - _de(p - wp.vec3(0.0, e, 0.0), power, iters)
    dz = _de(p + wp.vec3(0.0, 0.0, e), power, iters) - _de(p - wp.vec3(0.0, 0.0, e), power, iters)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _palette(t: float) -> wp.vec3:
    # IQ cosine palette — iridescent shells keyed on the orbit trap
    a = wp.vec3(0.5, 0.45, 0.4)
    b = wp.vec3(0.5, 0.5, 0.45)
    c = wp.vec3(1.0, 1.0, 1.0)
    d = wp.vec3(0.0, 0.15, 0.35)
    ph = (c * t + d) * _TWO_PI
    return a + wp.cw_mul(b, wp.vec3(wp.cos(ph[0]), wp.cos(ph[1]), wp.cos(ph[2])))


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, power: float, iters: int, steps: int) -> float:
    res = float(1.0)
    t = float(0.02)
    for _ in range(steps):
        h = _de(ro + rd * t, power, iters)
        if h < 0.0008:
            return 0.0
        res = wp.min(res, 9.0 * h / t)
        t += wp.clamp(h, 0.01, 0.2)
        if t > 4.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, power: float, iters: int) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.01 + 0.05 * float(k)
        d = _de(p + n * hr, power, iters)
        occ += (hr - d) * sca
        sca *= 0.8
    return wp.clamp(1.0 - 2.2 * occ, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  power: float, iters: int, spin: float, march_steps: int,
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
        v4 = mandelbulb_de(p, power, iters)
        d = v4[0]
        # closest-approach halo — weighted by the step size so it doesn't
        # pile up on grazing rays (that was blowing the whole bulb to white)
        step = wp.max(d * 0.82, 0.002)
        glow = glow + wp.exp(-d * 40.0) * step
        if d < 0.0006 * t + 0.0004:
            hit = 1
            trap = v4[1]
            break
        t += step
        if t > 6.0:
            break
    glow = wp.min(glow, 0.6)                             # hard cap the haze

    # dark-space background with a cool gradient
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.02, 0.03, 0.05) * (1.0 - up) + wp.vec3(0.04, 0.05, 0.09) * up

    if hit == 1:
        p = _roty(ro + rd * t, spin)
        n = _normal(p, power, iters)
        base = _palette(trap * 2.2 + 0.1)
        ndl = wp.max(wp.dot(n, sun), 0.0)
        sh = _soft_shadow(p + n * 0.004, sun, power, iters, shadow_steps)
        ao = _ao(p, n, power, iters)
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.5)
        # stronger key + AO so the fractal's crevices and shells read as detail
        col = wp.cw_mul(base, wp.vec3(0.10, 0.12, 0.18) * ao          # ambient
                        + wp.vec3(1.15, 1.05, 0.9) * (ndl * sh))       # sun
        col = col + base * (rim * 0.5)                                 # fresnel rim
        col = col * (0.35 + 0.65 * ao)                                 # deepen the pits

    # glow — a luminous haze bathing the fractal, tamed so it frames the bulb
    # instead of drowning it; brightest where the ray grazes empty space
    halo = wp.vec3(0.24, 0.42, 0.85) * (glow * 0.7)
    if hit == 1:
        halo = halo * 0.35                                            # not over the surface
    col = col + halo
    img[i, j] = col


def _tier_steps(name):
    return {"low": (90, 24, 14), "medium": (130, 36, 18), "high": (180, 50, 24),
            "ultra": (240, 70, 30)}.get(name, (130, 36, 18))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss, iters = _tier_steps(tier.name)
    power = 2.0 + 6.0 * (0.5 + 0.5 * math.sin(time * 0.25 - 1.5))     # 2..8 morph
    spin = time * 0.15 + float(mouse[0]) * 0.01
    az = 0.6
    el = 0.2 + float(mouse[1]) * 0.004
    dist = 2.9
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    sun = wp.normalize(wp.vec3(0.6, 0.5, 0.4))

    ssaa = 2                                            # fractals alias hard — SSAA
    W, H = int(width) * ssaa, int(height) * ssaa
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=42.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, sun, float(power), int(iters), float(spin),
                      int(ms), int(ss), int(W), int(H)], device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ssaa)
    r = max(2, int(min(width, height) * 0.014))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.4, radius=r, passes=3, octaves=3)
    out = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    return post.vignette(out, 0.3)


SCENE = Scene(
    name="mandelbulb",
    description="Distance-estimated Mandelbulb fractal (White-Nylander triplex "
                "power) — orbit-trap colour, glow, soft shadows; power morphs 2->8.",
    renderer=_render,
)
