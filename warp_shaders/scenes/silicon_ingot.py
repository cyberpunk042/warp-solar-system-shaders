"""Silicon ingot — a Czochralski monocrystalline boule.

The starting point of every chip: ultra-pure silicon is melted in a quartz
crucible and a seed crystal is dipped in and slowly pulled up while rotating, so
the melt solidifies onto it as a single continuous crystal — a mirror-bright grey
cylinder a metre long, rounded at the shoulders, tapering to the thin seed neck at
the top. This one turns on a studio stand; the whole industry is downstream of it.
See ``docs/research/35-electronics-components.md`` (Czochralski 1918).
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import (
    op_smooth_union, op_union, sd_cylinder, sd_sphere,
)
from ..scene import Scene

_MAXD = 30.0
_SIL = 0  # ec.SILICON


@wp.func
def _boule(p: wp.vec3) -> float:
    # main body: a tall capped cylinder, rounded shoulders
    body = sd_cylinder(p - wp.vec3(0.0, -0.1, 0.0), 1.35, 0.55)
    body = body - 0.05  # round the edges
    # bottom tail-cone (where the pull ended) — a squashed sphere blended on
    tail = sd_sphere(p - wp.vec3(0.0, -1.55, 0.0), 0.5)
    d = op_smooth_union(body, tail, 0.35)
    # seed neck at the very top: a thin cylinder tapering to the seed
    neck = sd_cylinder(p - wp.vec3(0.0, 1.55, 0.0), 0.28, 0.12)
    seed = sd_sphere(p - wp.vec3(0.0, 1.95, 0.0), 0.12)
    top = op_smooth_union(neck, seed, 0.12)
    d = op_smooth_union(d, top, 0.18)
    return d


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    # slow turntable rotation about Y
    a = time * 0.5
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    boule = _boule(q)
    floor = p[1] + 2.05
    return op_union(boule, floor)


@wp.func
def _mat_id(p: wp.vec3, time: float) -> int:
    if p[1] < -2.02:
        return 8  # dark glass floor
    return _SIL


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.02 + 0.12 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.2 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(160):
        p = eye + rd * t
        d = _map(p, time)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.9
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, time)
    m = _mat_id(p, time)
    ao = _ao(p, n, time)
    img[i, j] = ec.lit(n, rd, m, ao, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    az = 0.7 + float(mouse[0]) * 0.01
    el = 0.28 + float(mouse[1]) * 0.005
    dist = 6.4
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.3,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.15, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(40.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="silicon_ingot",
    description="a Czochralski monocrystalline silicon boule — the mirror-grey "
                "single crystal pulled from the melt, rounded shoulders tapering to "
                "the seed neck, turning on a studio stand. The source of all chips.",
    renderer=_render,
)
