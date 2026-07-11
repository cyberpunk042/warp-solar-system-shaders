"""Reflection + refraction showcase — the classic Whitted ray-tracing beauty shot.

A checkerboard floor with three spheres — a **silver mirror**, a **glass** ball,
and a **gold mirror** — that reflect the sky, the floor and *each other*. Because
Warp has no recursion, the specular path is an explicit **bounce loop**: at each
hit a mirror follows its reflection (`engine.raytrace.reflect`), the glass ball is
crossed by an analytic **double refraction** (Snell in, Snell out, with a Fresnel
rim reflection), and a matte hit terminates the path. Soft shadows + AO on the
floor, an analytic sky, and the host post pipeline.

`--quality` scales the march/shadow/AO steps; the bounce count is fixed at 4.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere
from ..engine.raytrace import fresnel_dielectric, reflect, refract
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.sdf import sd_sphere
from ..scene import Scene

_MAXD = 40.0
_IOR = wp.constant(1.5)
# the three spheres (centres + radius) — matched in _map / _mat_id / the glass path
_C1 = wp.constant(wp.vec3(-1.7, 0.0, 0.2))     # silver mirror
_C2 = wp.constant(wp.vec3(0.35, -0.1, 1.5))    # glass
_C3 = wp.constant(wp.vec3(1.8, 0.05, -0.3))    # gold mirror
_R = wp.constant(0.9)


@wp.func
def _map(p: wp.vec3) -> float:
    plane = p[1] + 1.0
    s = wp.min(sd_sphere(p - _C1, _R),
               wp.min(sd_sphere(p - _C2, _R), sd_sphere(p - _C3, _R)))
    return wp.min(plane, s)


@wp.func
def _mat_id(p: wp.vec3) -> int:
    d0 = p[1] + 1.0
    d1 = sd_sphere(p - _C1, _R)
    d2 = sd_sphere(p - _C2, _R)
    d3 = sd_sphere(p - _C3, _R)
    best = d0
    which = int(0)
    if d1 < best:
        best = d1
        which = 1
    if d2 < best:
        best = d2
        which = 2
    if d3 < best:
        best = d3
        which = 3
    return which


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _march(ro: wp.vec3, rd: wp.vec3, steps: int) -> float:
    t = float(0.01)
    for _ in range(steps):
        d = _map(ro + rd * t)
        if d < 0.0008 * t + 0.0003:
            return t
        t += d * 0.9
        if t > _MAXD:
            break
    return -1.0


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.03)
    for _ in range(steps):
        h = _map(ro + rd * t)
        if h < 0.001:
            return 0.0
        res = wp.min(res, 11.0 * h / t)
        t += wp.clamp(h, 0.02, 0.4)
        if t > 18.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.5, 0.62, 0.82) * (1.0 - up) + wp.vec3(0.11, 0.26, 0.62) * up
    s = wp.max(wp.dot(rd, sun), 0.0)
    return base + wp.vec3(1.0, 0.92, 0.75) * (wp.pow(s, 8.0) * 0.25 + wp.pow(s, 2000.0) * 14.0)


@wp.func
def _matte(p: wp.vec3, n: wp.vec3, mid: int, sun: wp.vec3, sh_steps: int) -> wp.vec3:
    # floor checker; the coloured/glossy fallbacks reuse this too
    chk = wp.floor(p[0] * 0.7) + wp.floor(p[2] * 0.7)
    g = 0.20 + 0.16 * wp.abs(chk - 2.0 * wp.floor(chk * 0.5))
    albedo = wp.vec3(g, g, g)
    ndl = wp.max(wp.dot(n, sun), 0.0)
    sh = _soft_shadow(p + n * 0.01, sun, sh_steps)
    amb = wp.cw_mul(_sky(n, sun), albedo) * 0.4
    return albedo * (ndl * sh) + amb


@wp.func
def _glass_exit(p: wp.vec3, rd_in: wp.vec3) -> wp.vec3:
    """Refract the interior ray out of the glass sphere (analytic far hit)."""
    h = ray_sphere(p + rd_in * 0.002, rd_in, _C2, _R)
    tex = h[1]
    if tex < 0.0:
        return rd_in
    pex = p + rd_in * (0.002 + tex)
    nex = wp.normalize(pex - _C2)                  # outward normal at exit
    return refract(rd_in, nex, _IOR)               # glass -> air (may TIR)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  march_steps: int, shadow_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = wp.vec3(0.0, 0.0, 0.0)
    tp = wp.vec3(1.0, 1.0, 1.0)                     # throughput

    for _bounce in range(4):
        t = _march(ro, rd, march_steps)
        if t < 0.0:
            col = col + wp.cw_mul(tp, _sky(rd, sun))
            break
        p = ro + rd * t
        n = _normal(p)
        mid = _mat_id(p)

        if mid == 1 or mid == 3:                    # mirror (silver / gold)
            tint = wp.vec3(0.93, 0.94, 0.96)
            if mid == 3:
                tint = wp.vec3(1.0, 0.80, 0.38)     # gold
            col = col + wp.cw_mul(tp, _matte(p, n, mid, sun, shadow_steps) * 0.05)
            tp = wp.cw_mul(tp, tint)
            rd = reflect(rd, n)
            ro = p + n * 0.004
        elif mid == 2:                              # glass: double refraction
            ci = wp.max(wp.dot(-rd, n), 0.0)
            fr = fresnel_dielectric(ci, _IOR)
            rd_in = refract(rd, n, 1.0 / _IOR)
            rd_out = _glass_exit(p, rd_in)
            col = col + wp.cw_mul(tp, _sky(reflect(rd, n), sun) * (fr * 0.4))
            tp = wp.cw_mul(tp, wp.vec3(0.90, 0.97, 0.94) * (1.0 - fr * 0.35))
            rd = rd_out
            ro = p + rd_out * 0.02
        else:                                       # floor: matte, terminal
            col = col + wp.cw_mul(tp, _matte(p, n, mid, sun, shadow_steps))
            break

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = active_tier()
    q = make_quality_steps(tier.name)
    az = 0.7 + 0.2 * math.sin(time * 0.2) + float(mouse[0]) * 0.01
    el = 0.28 + float(mouse[1]) * 0.004
    dist = 6.0
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el) + 0.7,
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, -0.15, 0.4), fov_deg=44.0, aspect=width / height)
    sa = 0.7 + 0.2 * math.sin(time * 0.3)
    sun = wp.vec3(math.cos(sa) * 0.5, 0.72, math.sin(sa) * 0.45)
    sun = wp.normalize(sun)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(q[0]), int(q[1]), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.4, radius=r, passes=2)
    out = post.tonemap(hdr, mode="aces", exposure=1.1)
    return post.vignette(out, 0.26)


def make_quality_steps(name):
    return {"low": (72, 20), "medium": (110, 30), "high": (160, 44),
            "ultra": (220, 60)}.get(name, (110, 30))


SCENE = Scene(
    name="reflections",
    description="Whitted ray-tracing showcase: mirror + glass + gold spheres on a "
                "checkerboard, reflecting each other (engine.raytrace bounce loop).",
    renderer=_render,
)
