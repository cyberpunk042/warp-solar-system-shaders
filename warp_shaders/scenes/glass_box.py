"""Glass box — path-traced reflection and refraction inside a lit room.

The engine leap continued: the same Monte-Carlo path tracer as `cornell_box`, but now
with **specular materials**. A room (red left, green right) lit by a ceiling patch holds a
**glass sphere** — light refracts through it (Snell's law) and glints off it (Fresnel) — and
a **mirror sphere** that reflects the whole coloured room. Diffuse walls still bounce and
bleed colour as before, so reflection, refraction, and global illumination all share one
unbiased integrator. See ``docs/research/39-engine-leap.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.sdf import op_union, sd_box, sd_sphere
from ..scene import Scene

_RX = 1.0
_RY = 1.0
_RZ = 1.0
_SPP = 384
_BOUNCES = 6
_GLASS = wp.constant(wp.vec3(-0.42, -0.55, -0.2))     # glass sphere centre
_MIRROR = wp.constant(wp.vec3(0.45, -0.62, 0.3))      # mirror sphere centre
_GR = 0.42
_MR = 0.36


@wp.func
def _walls(p: wp.vec3) -> float:
    th = 0.04
    floor = sd_box(p - wp.vec3(0.0, -_RY, 0.0), wp.vec3(_RX, th, _RZ))
    ceil = sd_box(p - wp.vec3(0.0, _RY, 0.0), wp.vec3(_RX, th, _RZ))
    back = sd_box(p - wp.vec3(0.0, 0.0, -_RZ), wp.vec3(_RX, _RY, th))
    left = sd_box(p - wp.vec3(-_RX, 0.0, 0.0), wp.vec3(th, _RY, _RZ))
    right = sd_box(p - wp.vec3(_RX, 0.0, 0.0), wp.vec3(th, _RY, _RZ))
    return op_union(op_union(floor, ceil), op_union(back, op_union(left, right)))


@wp.func
def _scene(p: wp.vec3) -> float:
    g = sd_sphere(p - _GLASS, _GR)
    m = sd_sphere(p - _MIRROR, _MR)
    return op_union(_walls(p), op_union(g, m))


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0015
    dx = _scene(p + wp.vec3(e, 0.0, 0.0)) - _scene(p - wp.vec3(e, 0.0, 0.0))
    dy = _scene(p + wp.vec3(0.0, e, 0.0)) - _scene(p - wp.vec3(0.0, e, 0.0))
    dz = _scene(p + wp.vec3(0.0, 0.0, e)) - _scene(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _matid(p: wp.vec3) -> int:
    # 0 diffuse, 1 mirror, 2 glass
    if wp.length(p - _GLASS) < _GR + 0.02:
        return 2
    if wp.length(p - _MIRROR) < _MR + 0.02:
        return 1
    return 0


@wp.func
def _albedo(p: wp.vec3) -> wp.vec3:
    if p[0] < -0.9:
        return wp.vec3(0.62, 0.06, 0.06)
    if p[0] > 0.9:
        return wp.vec3(0.10, 0.48, 0.10)
    return wp.vec3(0.72, 0.72, 0.72)


@wp.func
def _emission(p: wp.vec3) -> wp.vec3:
    if p[1] > 0.9 and wp.abs(p[0]) < 0.4 and wp.abs(p[2]) < 0.4:
        return wp.vec3(18.0, 15.0, 11.0)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _march(ro: wp.vec3, rd: wp.vec3):
    t = float(0.004)
    hit = int(0)
    for _ in range(110):
        p = ro + rd * t
        d = _scene(p)
        if d < 0.0006:
            hit = 1
            break
        t += wp.max(d * 0.85, 0.0008)
        if t > 14.0:
            break
    return t, hit


@wp.func
def _onb(n: wp.vec3, r1: float, r2: float) -> wp.vec3:
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    tang = wp.normalize(wp.cross(a, n))
    bit = wp.cross(n, tang)
    r = wp.sqrt(r1)
    phi = 6.2831853 * r2
    return wp.normalize(tang * (r * wp.cos(phi)) + bit * (r * wp.sin(phi)) + n * wp.sqrt(wp.max(0.0, 1.0 - r1)))


@wp.func
def _fresnel(cosi: float, ior: float) -> float:
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    x = 1.0 - cosi
    return r0 + (1.0 - r0) * x * x * x * x * x


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanfov: float,
                   spp: int, seed0: int):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    acc = wp.vec3(0.0, 0.0, 0.0)
    pix = i * width + j

    for s in range(spp):
        rng = wp.rand_init(seed0 + pix * 9781 + s * 6151, s * 2749 + 17)
        du = wp.randf(rng) - 0.5
        dv = wp.randf(rng) - 0.5
        u = (2.0 * (float(j) + 0.5 + du) / float(width) - 1.0) * tanfov * aspect
        v = (2.0 * (float(height - 1 - i) + 0.5 + dv) / float(height) - 1.0) * tanfov
        rd = wp.normalize(fwd + right * u + up * v)
        ro = eye
        throughput = wp.vec3(1.0, 1.0, 1.0)
        radiance = wp.vec3(0.0, 0.0, 0.0)

        for _b in range(_BOUNCES):
            t, hit = _march(ro, rd)
            if hit == 0:
                break
            p = ro + rd * t
            ng = _normal(p)
            mat = _matid(p)
            emit = _emission(p)
            if emit[0] > 0.0:
                radiance = radiance + wp.cw_mul(throughput, emit)
                break
            if mat == 1:                                        # mirror
                n = ng
                if wp.dot(n, rd) > 0.0:
                    n = -n
                rd = rd - n * (2.0 * wp.dot(rd, n))
                throughput = wp.cw_mul(throughput, wp.vec3(0.95, 0.95, 0.97))
                ro = p + n * 0.002
            elif mat == 2:                                      # glass (dielectric)
                outward = ng
                cosi = wp.dot(rd, ng)
                ior = float(1.5)
                if cosi < 0.0:                                  # entering
                    eta = 1.0 / ior
                    ci = -cosi
                    n = ng
                else:                                           # exiting
                    eta = ior
                    ci = cosi
                    n = -ng
                k = 1.0 - eta * eta * (1.0 - ci * ci)
                fr = _fresnel(ci, ior)
                if k < 0.0 or wp.randf(rng) < fr:               # reflect (or TIR)
                    rd = rd - n * (2.0 * wp.dot(rd, n))
                    ro = p + n * 0.002
                else:                                           # refract
                    rd = wp.normalize(rd * eta + n * (eta * ci - wp.sqrt(k)))
                    ro = p - n * 0.002
                # (clear glass — no absorption tint)
            else:                                               # diffuse
                n = ng
                if wp.dot(n, rd) > 0.0:
                    n = -n
                throughput = wp.cw_mul(throughput, _albedo(p))
                ro = p + n * 0.002
                rd = _onb(n, wp.randf(rng), wp.randf(rng))
        acc = acc + radiance

    img[i, j] = acc / float(spp)


def _render(width, height, time, mouse, device):
    spp = _SPP
    if width * height <= 96 * 72:
        spp = 8

    eye = wp.vec3(0.0 + float(mouse[0]) * 0.004, 0.05, 2.75)
    tgt = wp.vec3(0.0, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov, spp, 22222],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.5, preserve_hue=True)


SCENE = Scene(
    name="glass_box",
    description="a path-traced room with specular materials — a glass sphere refracting and "
                "glinting (Snell + Fresnel) and a mirror sphere reflecting the red/green "
                "room, with diffuse global illumination bleeding colour. Reflection, "
                "refraction and GI in one unbiased path tracer.",
    renderer=_render,
)
